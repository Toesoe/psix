"""psix — Pakon Scan Interface for Linux.

Flask app factory.  The web framework + visual design are reused from piPalette
(an unrelated Polaroid film-recorder app) purely as a template — same vendor
look, no shared domain.

Ships as a native desktop window via pywebview (see ../main.py); the same app
also runs as a plain local server for development.
"""

import json
import queue
import shutil
import time
from pathlib import Path

from flask import (
    Flask, Response, abort, jsonify, redirect, render_template, request,
    send_file, url_for,
)

from .config import Config, user_data_dir
from .driver import ScannerDriver
from .processing import (
    DEFAULT_GRADE, ICE_DEFAULTS as DEFAULT_ICE, export_frame, grade_preview_jpeg,
    grade_with_defaults, ice_preview_jpeg, ice_view_jpeg,
)
from .rolls import RollStore
from .scan_runner import ScanBusyError, ScanRunner
from .scanner import ScannerManager


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _coerce_bool(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _grade_from_payload(payload):
    """Pick + float-coerce the known grade keys from a request payload."""
    grade = {}
    for key in DEFAULT_GRADE:
        if key in payload:
            try:
                grade[key] = float(payload[key])
            except (TypeError, ValueError):
                pass
    return grade


def _format_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}" if n < 100 else f"{n:.0f} {unit}"
        n /= 1024


def _storage_snapshot(output_dir):
    """Disk-free stats for the sidebar, anchored on the output directory."""
    probe = Path(output_dir)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return None
    used_pct = (usage.used / usage.total) if usage.total else 0.0
    level = "danger" if used_pct >= 0.90 else "warn" if used_pct >= 0.70 else "ok"
    return {
        "disk_free_label": _format_bytes(usage.free),
        "disk_total_label": _format_bytes(usage.total),
        "level": level,
    }


def create_app(data_dir=None):
    pkg_dir = Path(__file__).resolve().parent                 # the psix package (templates/static live here)
    data_dir = Path(data_dir) if data_dir else user_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(pkg_dir / "templates"),
        static_folder=str(pkg_dir / "static"),
    )

    config = Config(data_dir / "config.json")
    driver = ScannerDriver()
    rolls = RollStore(config)
    runner = ScanRunner(config, rolls, driver)
    # The scanner monitor publishes 'device' events on the runner's SSE bus.
    scanner = ScannerManager(config, driver, publish=runner._publish)
    runner.bind_monitor(scanner.monitor)
    scanner.start()

    app.config["PSIX_CONFIG"] = config
    app.config["PSIX_SCANNER"] = scanner
    app.config["PSIX_ROLLS"] = rolls
    app.config["PSIX_RUNNER"] = runner

    @app.context_processor
    def inject_globals():
        return {
            "storage": _storage_snapshot(config.get("output_dir")),
            "output_dir": config.get("output_dir"),
        }

    # ---- pages -----------------------------------------------------------

    @app.route("/")
    def index():
        return redirect(url_for("scan_page"))

    @app.route("/scan")
    def scan_page():
        return render_template(
            "scan.html",
            view="scan",
            status=scanner.status(),
            rolls=rolls.list_rolls(),
            next_label=rolls.next_label(),
            run=runner.state(),
        )

    @app.route("/rolls/<int:roll_id>")
    def roll_detail_page(roll_id):
        roll = rolls.get(roll_id)
        if roll is None:
            abort(404)
        frames = rolls.frames(roll)
        for f in frames:                                 # merge each frame's grade with defaults
            f["grade"] = grade_with_defaults(f.get("grade"))
        initial = frames[0]["grade"] if frames else grade_with_defaults(None)
        return render_template(
            "roll_detail.html",
            view="scan",
            status=scanner.status(),
            roll=roll,
            frames=frames,
            grade=initial,                               # sliders start on the first frame's grade
            default_grade=grade_with_defaults(None),     # neutral grade, for the Reset button
            ice={**DEFAULT_ICE, **(roll.get("ice") or {})},   # roll's committed ICE params (or defaults)
            has_ir=rolls.has_ir(roll),
        )

    @app.route("/rolls/<int:roll_id>/preview/<path:filename>")
    def roll_preview(roll_id, filename):
        roll = rolls.get(roll_id)
        if roll is None:
            abort(404)
        path = rolls.preview_path(roll, filename)
        if path is None:
            abort(404)
        # mimetype inferred from the extension (.jpg real frames / .svg mock).
        return send_file(path, max_age=0)

    @app.route("/settings")
    def settings_page():
        return render_template(
            "settings.html",
            view="settings",
            status=scanner.status(),
            config=config.all(),
        )

    @app.route("/partials/topbar")
    def partial_topbar():
        return render_template("partials/topbar.html", status=scanner.status())

    @app.route("/partials/scanner-info")
    def partial_scanner_info():
        return render_template("partials/scanner_info.html", status=scanner.status())

    # ---- scanner / config APIs ------------------------------------------

    @app.route("/api/status")
    def api_status():
        return jsonify(scanner.status())

    @app.route("/api/scanner/connect", methods=["POST"])
    def api_scanner_connect():
        return jsonify(scanner.connect())

    @app.route("/api/firmware")
    def api_firmware():
        return jsonify(driver.firmware_status())

    @app.route("/api/firmware/upload", methods=["POST"])
    def api_firmware_upload():
        files = request.files.getlist("firmware")
        if not files:
            return jsonify({"error": "no files"}), 400
        saved, errors = [], []
        for f in files:
            try:
                status = driver.save_firmware(f.filename, f.read())
                saved.append(f.filename)
            except ValueError as exc:
                errors.append({"file": f.filename, "error": str(exc)})
        if saved:
            scanner.monitor.force()        # re-check now: load firmware if it's complete
        code = 200 if saved else 400
        return jsonify({"saved": saved, "errors": errors,
                        "firmware": driver.firmware_status()}), code

    @app.route("/api/config", methods=["POST"])
    def api_config():
        payload = request.get_json(silent=True) or request.form.to_dict()
        cleaned = {}
        if "mock_mode" in payload:
            cleaned["mock_mode"] = _coerce_bool(payload["mock_mode"])
        if "auto_prepare" in payload:
            cleaned["auto_prepare"] = _coerce_bool(payload["auto_prepare"])
        if "output_dir" in payload:
            out = (payload["output_dir"] or "").strip()
            if out:
                cleaned["output_dir"] = out
        config.update(**cleaned)
        return jsonify({"config": config.all(), "status": scanner.status()})

    # ---- rolls -----------------------------------------------------------

    @app.route("/api/rolls")
    def api_rolls():
        return jsonify({"rolls": rolls.list_rolls(), "next_label": rolls.next_label()})

    @app.route("/api/rolls", methods=["POST"])
    def api_rolls_create():
        payload = request.get_json(silent=True) or request.form.to_dict()
        roll = rolls.create_roll(payload.get("name"))
        return jsonify({"roll": roll}), 201

    @app.route("/api/rolls/<int:roll_id>/scan", methods=["POST"])
    def api_roll_scan(roll_id):
        payload = request.get_json(silent=True) or {}
        ir = _coerce_bool(payload.get("ir", False))
        try:
            out_path = runner.start_scan(roll_id, ir=ir)
        except ScanBusyError as exc:
            return jsonify({"error": str(exc)}), 409
        except KeyError:
            return jsonify({"error": "roll not found"}), 404
        return jsonify({"started": roll_id, "out_path": out_path}), 202

    @app.route("/api/rolls/<int:roll_id>/process", methods=["POST"])
    def api_roll_process(roll_id):
        """Generate (or regenerate) preview frames for a roll's captures."""
        try:
            runner.process_roll(roll_id)
        except ScanBusyError as exc:
            return jsonify({"error": str(exc)}), 409
        except KeyError:
            return jsonify({"error": "roll not found"}), 404
        return jsonify({"processing": roll_id}), 202

    @app.route("/api/rolls/<int:roll_id>/frame/<int:idx>/grade", methods=["POST"])
    def api_frame_grade(roll_id, idx):
        """Persist one frame's grade and re-render just that frame (full quality)."""
        grade = _grade_from_payload(request.get_json(silent=True) or {})
        try:
            runner.apply_frame_grade(roll_id, idx, grade)
        except ScanBusyError as exc:
            return jsonify({"error": str(exc)}), 409
        except KeyError:
            return jsonify({"error": "roll not found"}), 404
        except IndexError:
            return jsonify({"error": "frame out of range"}), 404
        return jsonify({"grading": [roll_id, idx]}), 202

    @app.route("/api/rolls/<int:roll_id>/frame/<int:idx>/preview", methods=["POST"])
    def api_frame_preview(roll_id, idx):
        """Live-slider preview: grade ONE frame's downsized negative → JPEG.
        Synchronous + transient (not persisted); the client swaps the centre image."""
        roll = rolls.get(roll_id)
        if roll is None:
            abort(404)
        negs = rolls.frame_negs(roll)
        if idx < 0 or idx >= len(negs) or not negs[idx].exists():
            return jsonify({"error": "no cached negative for this frame"}), 404
        grade = _grade_from_payload(request.get_json(silent=True) or {})
        try:
            data = grade_preview_jpeg(driver, str(negs[idx]), grade)
        except Exception as exc:                          # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/rolls/<int:roll_id>/apply_ice", methods=["POST"])
    def api_apply_ice(roll_id):
        """Commit a chosen ICE detection: re-develop the roll's IR negatives at the
        posted threshold/kernel and persist them on the roll."""
        payload = request.get_json(silent=True) or {}
        try:
            thr = float(payload.get("ir_thresh", DEFAULT_ICE["ir_thresh"]))
            ker = int(float(payload.get("ir_kernel", DEFAULT_ICE["ir_kernel"])))
            mns = int(float(payload.get("ir_min_size", DEFAULT_ICE["ir_min_size"])))
        except (TypeError, ValueError):
            return jsonify({"error": "bad ICE params"}), 400
        ker = max(3, ker | 1)
        mns = max(1, mns)
        try:
            runner.apply_ice(roll_id, thr, ker, mns)
        except ScanBusyError as exc:
            return jsonify({"error": str(exc)}), 409
        except KeyError:
            return jsonify({"error": "roll not found"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"applying_ice": roll_id, "ir_thresh": thr, "ir_kernel": ker,
                        "ir_min_size": mns}), 202

    @app.route("/api/rolls/<int:roll_id>/frame/<int:idx>/export", methods=["POST"])
    def api_frame_export(roll_id, idx):
        """Export ONE frame at full resolution into <roll>/export/, using the grade in the
        request (the on-screen sliders), falling back to the frame's stored grade."""
        roll = rolls.get(roll_id)
        if roll is None:
            abort(404)
        frames = rolls.frames(roll)
        negs = rolls.frame_negs(roll)
        if idx < 0 or idx >= len(frames) or idx >= len(negs):
            return jsonify({"error": "frame out of range"}), 404
        neg = negs[idx]
        if not neg.exists():
            return jsonify({"error": "no negative for this frame (export needs a real scan)"}), 404
        g = dict((roll.get("frame_grades") or {}).get(frames[idx]["filename"]) or {})
        g.update(_grade_from_payload(request.get_json(silent=True) or {}))
        name = "%s_%s.jpg" % (roll["label"], frames[idx]["label"])
        try:
            export_frame(driver, str(neg), rolls.export_dir(roll) / name, grade_with_defaults(g))
        except Exception as exc:                          # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        return jsonify({"exported": name, "dir": "export"}), 200

    @app.route("/api/rolls/<int:roll_id>/frame/<int:idx>/ice_view", methods=["POST"])
    def api_frame_ice_view(roll_id, idx):
        """ICE interface: graded RGB positive with ICE on/off (de-dusted vs original) and an
        optional live red detection overlay. Transient (not persisted)."""
        roll = rolls.get(roll_id)
        if roll is None:
            abort(404)
        negs = rolls.frame_negs(roll)
        srcs = rolls.frame_ir_sources(roll)
        if idx < 0 or idx >= len(negs) or idx >= len(srcs):
            return jsonify({"error": "frame out of range"}), 404
        neg = negs[idx]
        ir_plane, flatref = srcs[idx]
        if not neg.exists() or not ir_plane.exists() or flatref is None or not flatref.exists():
            return jsonify({"error": "no IR data for this frame (not a 4-channel scan)"}), 404
        ice_npz = neg.with_name(neg.name.replace("_neg.tiff", "_neg_ice.npz"))
        payload = request.get_json(silent=True) or {}
        grade = _grade_from_payload(payload)
        ice_on = _coerce_bool(payload.get("ice_on", True))
        show_mask = _coerce_bool(payload.get("show_mask", False))
        try:
            thr = float(payload.get("ir_thresh", DEFAULT_ICE["ir_thresh"]))
            ker = max(3, int(float(payload.get("ir_kernel", DEFAULT_ICE["ir_kernel"]))) | 1)
            mns = max(1, int(float(payload.get("ir_min_size", DEFAULT_ICE["ir_min_size"]))))
        except (TypeError, ValueError):
            thr, ker, mns = DEFAULT_ICE["ir_thresh"], DEFAULT_ICE["ir_kernel"], DEFAULT_ICE["ir_min_size"]
        try:
            data = ice_view_jpeg(driver, str(neg), str(ir_plane), str(flatref), str(ice_npz),
                                 grade, ice_on, show_mask, thr, ker, mns)
        except Exception as exc:                          # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/rolls/<int:roll_id>/frame/<int:idx>/ice_preview", methods=["POST"])
    def api_frame_ice_preview(roll_id, idx):
        """Live ICE dust-preview: render the frame's IR channel with detected
        dust/scratches overlaid in red, at the posted detection params. Transient."""
        roll = rolls.get(roll_id)
        if roll is None:
            abort(404)
        srcs = rolls.frame_ir_sources(roll)
        if idx < 0 or idx >= len(srcs):
            return jsonify({"error": "frame out of range"}), 404
        ir_plane, flatref = srcs[idx]
        if not ir_plane.exists() or flatref is None or not flatref.exists():
            return jsonify({"error": "no IR plane for this frame (not a 4-channel scan)"}), 404
        payload = request.get_json(silent=True) or {}
        try:
            thr = float(payload.get("ir_thresh", DEFAULT_ICE["ir_thresh"]))
            ker = int(float(payload.get("ir_kernel", DEFAULT_ICE["ir_kernel"])))
            mns = int(float(payload.get("ir_min_size", DEFAULT_ICE["ir_min_size"])))
        except (TypeError, ValueError):
            thr, ker, mns = DEFAULT_ICE["ir_thresh"], DEFAULT_ICE["ir_kernel"], DEFAULT_ICE["ir_min_size"]
        ker = max(3, ker | 1)                            # odd, >=3
        mns = max(1, mns)
        try:
            data = ice_preview_jpeg(driver, str(ir_plane), str(flatref), thr, ker, mns)
        except Exception as exc:                          # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/scan", methods=["POST"])
    def api_scan_new():
        """One-shot: create a new roll and immediately start scanning into it."""
        payload = request.get_json(silent=True) or request.form.to_dict()
        ir = _coerce_bool(payload.get("ir", False))
        roll = rolls.create_roll(payload.get("name"))
        try:
            out_path = runner.start_scan(roll["id"], ir=ir)
        except ScanBusyError as exc:
            return jsonify({"error": str(exc), "roll": roll}), 409
        return jsonify({"roll": roll, "started": roll["id"], "out_path": out_path}), 202

    # ---- scan runner / SSE ----------------------------------------------

    @app.route("/api/scan/state")
    def api_scan_state():
        return jsonify(runner.state())

    @app.route("/api/device/state")
    def api_device_state():
        return jsonify(scanner.status())

    @app.route("/api/scan/stop", methods=["POST"])
    def api_scan_stop():
        runner.stop_scan()
        return jsonify({"stopping": True}), 202

    @app.route("/api/events")
    def api_events():
        """Unified SSE: scan lifecycle ('state'/'done'/'error') + device
        presence/readiness ('device'). Opened on every page."""
        def stream():
            sub = runner.subscribe()
            try:
                yield _sse("state", runner.state())
                yield _sse("device", scanner.status())
                while True:
                    try:
                        event, data = sub.get(timeout=15)
                        yield _sse(event, data)
                    except queue.Empty:
                        yield ": keepalive\n\n"
            except GeneratorExit:
                pass
            finally:
                runner.unsubscribe(sub)

        resp = Response(stream(), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    return app
