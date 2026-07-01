// psix — vanilla JS frontend (no build, no deps). Sibling to piPalette.

(function () {
  "use strict";

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

  function toast(message, kind) {
    var stack = $("#toast-stack");
    if (!stack) return;
    var el = document.createElement("div");
    el.className = "toast" + (kind ? " toast-" + kind : "");
    el.textContent = message;
    stack.appendChild(el);
    setTimeout(function () {
      el.style.transition = "opacity 200ms ease";
      el.style.opacity = "0";
      setTimeout(function () { el.remove(); }, 250);
    }, 3200);
  }

  async function jsonFetch(url, options) {
    options = options || {};
    if (options.body && typeof options.body === "object" && !(options.body instanceof FormData)) {
      options.headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
      options.body = JSON.stringify(options.body);
    }
    var res = await fetch(url, options);
    var ct = res.headers.get("content-type") || "";
    var body = ct.indexOf("application/json") >= 0 ? await res.json() : await res.text();
    if (!res.ok) {
      var msg = (body && body.error) || (typeof body === "string" ? body : res.statusText);
      throw new Error(msg);
    }
    return body;
  }

  async function htmlFetch(url) {
    var res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load " + url);
    return res.text();
  }

  async function refreshTopbar() {
    try {
      var html = await htmlFetch("/partials/topbar");
      var head = $(".topbar");
      if (head) head.innerHTML = html;
    } catch (err) { console.warn(err); }
  }

  // -------- scanner connect -------------------------------------------

  async function connectScanner() {
    toast("Connecting to scanner…");
    try {
      var status = await jsonFetch("/api/scanner/connect", { method: "POST" });
      await refreshTopbar();
      var info = $("#scanner-info");
      if (info) info.innerHTML = await htmlFetch("/partials/scanner-info");
      if (status.connected) {
        toast(status.mock_mode ? "Mock scanner ready" : "Scanner connected", "ok");
      } else {
        toast(status.error || "No scanner found", "err");
      }
    } catch (err) {
      toast("Connect failed: " + err.message, "err");
    }
  }

  // -------- settings: config auto-save --------------------------------

  async function saveConfig(changes) {
    try {
      await jsonFetch("/api/config", { method: "POST", body: changes });
      toast("Saved", "ok");
      await refreshTopbar();
      var info = $("#scanner-info");
      if (info) info.innerHTML = await htmlFetch("/partials/scanner-info");
    } catch (err) {
      toast("Save failed: " + err.message, "err");
    }
  }

  function wireConfigForms() {
    $$("[data-config-form]").forEach(function (form) {
      // Mode radios save immediately on change.
      $$('input[name="mock_mode"]', form).forEach(function (radio) {
        radio.addEventListener("change", function () {
          saveConfig({ mock_mode: radio.value });
          // reload so the hint text + scanner card reflect the new mode
          setTimeout(function () { location.reload(); }, 250);
        });
      });
      // Checkboxes (e.g. auto_prepare) save immediately as booleans.
      $$('input[type="checkbox"]', form).forEach(function (cb) {
        cb.addEventListener("change", function () {
          var o = {}; o[cb.name] = cb.checked;
          saveConfig(o);
        });
      });
      // Text inputs save on blur / Enter.
      $$('input[type="text"], input.input', form).forEach(function (input) {
        if (input.name === "mock_mode") return;
        function commit() {
          var o = {}; o[input.name] = input.value;
          saveConfig(o);
        }
        input.addEventListener("blur", commit);
        input.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
        });
      });
    });
  }

  // -------- scan: start + live progress (SSE) -------------------------

  function fmtBytes(n) {
    n = Number(n) || 0;
    var u = ["B", "KB", "MB", "GB"];
    var i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n : n.toFixed(1)) + " " + u[i];
  }

  function renderRun(state) {
    var bar = $("#run-status");
    if (!bar) return;
    var busy = state.busy;
    var phase = state.phase || "idle";
    bar.hidden = !busy && phase === "idle";
    bar.classList.toggle("run-done", phase === "done");
    bar.classList.toggle("run-failed", phase === "failed");
    bar.classList.toggle("run-stopping", phase === "stopping");
    var label = state.roll_label ? ("Roll " + state.roll_label + " — ") : "";
    $("#run-status-text").textContent = label + (state.message || phase);
    $("#run-status-progress").textContent = state.bytes ? fmtBytes(state.bytes) : "";
    var btn = $("#stop-scan-btn");
    if (btn) btn.hidden = !busy;
    var start = $("#start-scan-btn");
    if (start) start.disabled = busy;
  }

  // -------- live device status (every page) ---------------------------

  var lastDevicePhase = null;

  async function updateDeviceUI(dev) {
    // Refresh the topbar pill + (on Settings) the identity card from the server.
    try {
      var head = $(".topbar");
      if (head) head.innerHTML = await htmlFetch("/partials/topbar");
    } catch (e) {}
    var info = $("#scanner-info");
    if (info) { try { info.innerHTML = await htmlFetch("/partials/scanner-info"); } catch (e) {} }

    // Toast only on meaningful transitions (not mock idle).
    if (dev && dev.phase && dev.phase !== lastDevicePhase) {
      var prev = lastDevicePhase;
      lastDevicePhase = dev.phase;
      if (prev !== null) {
        if (dev.phase === "ready" && !dev.mock_mode) toast("Scanner ready", "ok");
        else if (dev.phase === "firmware_missing") toast("Scanner firmware needed — open Settings to add it", "warn");
        else if (dev.phase === "loading") toast("Loading scanner firmware…");
        else if (dev.phase === "preparing") toast("Initializing scanner…");
        else if (dev.phase === "fault") toast(dev.error || "Scanner error", "err");
        else if (dev.phase === "absent" && !dev.mock_mode) toast("Scanner disconnected", "warn");
      }
    }
  }

  // -------- unified event stream (scan + device), opened on every page -

  function connectEventStream() {
    var es = new EventSource("/api/events");
    es.addEventListener("state", function (e) {
      if ($("#run-status")) renderRun(JSON.parse(e.data));
    });
    es.addEventListener("done", function (e) {
      var d = {};
      try { d = JSON.parse(e.data); } catch (x) {}
      if (typeof d.frame === "number") {            // per-frame grade applied
        refreshFrame(d.frame);
        toast("Grade applied", "ok");
        return;
      }
      if (!$("#run-status")) return;                // scan / develop completion
      toast(d.bytes ? ("Scan complete — " + fmtBytes(d.bytes)) : "Done", "ok");
      setTimeout(function () { location.reload(); }, 600);
    });
    es.addEventListener("error", function (e) {
      if (e.data) { try { toast("Scan failed: " + JSON.parse(e.data).error, "err"); } catch (x) {} }
    });
    es.addEventListener("device", function (e) {
      try { updateDeviceUI(JSON.parse(e.data)); } catch (x) {}
    });
  }

  async function startNewScan(name, ir) {
    try {
      var data = await jsonFetch("/api/scan", { method: "POST", body: { name: name, ir: ir } });
      toast("Started Roll " + data.roll.label, "ok");
    } catch (err) {
      toast("Could not start: " + err.message, "err");
    }
  }

  async function scanIntoRoll(rollId, ir) {
    try {
      await jsonFetch("/api/rolls/" + rollId + "/scan", { method: "POST", body: { ir: ir } });
      toast("Scan started", "ok");
    } catch (err) {
      toast("Could not start: " + err.message, "err");
    }
  }

  function wireScanPage() {
    var form = $("#new-scan-form");
    if (form) {
      form.addEventListener("submit", function (ev) {
        ev.preventDefault();
        var name = $("#roll-name").value;
        var ir = $("#ir-toggle") && $("#ir-toggle").checked;
        startNewScan(name, ir);
      });
    }
    var stop = $("#stop-scan-btn");
    if (stop) {
      stop.addEventListener("click", function () {
        jsonFetch("/api/scan/stop", { method: "POST" }).catch(function () {});
      });
    }
  }

  // -------- live colour-grade preview (centre frame) ------------------

  var live = { rollId: null, idx: 0, dirty: false, timer: null, ctrl: null, url: null };
  var ice = { on: true, mask: false, timer: null, ctrl: null, url: null };

  function livePreview() {
    if (typeof iceViewActive === "function" && iceViewActive()) { renderIceView(); return; }
    var form = $("#grade-form");
    var img = $("#stage-img");
    if (!form || !img || live.rollId == null) return;
    var grade = currentGrade();
    if (live.ctrl) live.ctrl.abort();
    live.ctrl = ("AbortController" in window) ? new AbortController() : null;
    fetch("/api/rolls/" + live.rollId + "/frame/" + live.idx + "/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(grade),
      signal: live.ctrl ? live.ctrl.signal : undefined,
    }).then(function (r) {
      if (!r.ok) throw new Error("preview " + r.status);
      return r.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob);
      img.src = url;
      if (live.url) URL.revokeObjectURL(live.url);
      live.url = url;
    }).catch(function () { /* aborted/failed: keep current image */ });
  }

  function scheduleLive() {
    clearTimeout(live.timer);
    live.timer = setTimeout(livePreview, 120);
  }

  // -------- ICE: two eye toggles on the RGB preview -------------------
  // ice.on   = de-dusted (true) vs original/dusty (false)  -> before/after
  // ice.mask = overlay the live red detection on the RGB

  function iceParams() {
    var t = $("#ice-thresh"), k = $("#ice-kernel"), m = $("#ice-min-size");
    return {
      ir_thresh: t ? parseFloat(t.value) : 0.04,
      ir_kernel: k ? parseInt(k.value, 10) : 41,
      ir_min_size: m ? parseInt(m.value, 10) : 3,
    };
  }

  function iceViewActive() { return (!ice.on) || ice.mask; }   // needs a server render vs plain committed RGB

  function renderIceView() {
    var img = $("#stage-img");
    if (!img || live.rollId == null) return;
    if (ice.ctrl) ice.ctrl.abort();
    ice.ctrl = ("AbortController" in window) ? new AbortController() : null;
    var body = Object.assign({}, currentGrade(), iceParams(), { ice_on: ice.on, show_mask: ice.mask });
    fetch("/api/rolls/" + live.rollId + "/frame/" + live.idx + "/ice_view", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ice.ctrl ? ice.ctrl.signal : undefined,
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || r.status); });
      return r.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob);
      img.src = url;
      if (ice.url) URL.revokeObjectURL(ice.url);
      ice.url = url;
    }).catch(function (err) {
      if (err && err.name !== "AbortError") toast("ICE: " + err.message, "err");
    });
  }

  function scheduleIceView() {
    clearTimeout(ice.timer);
    ice.timer = setTimeout(renderIceView, 120);
  }

  function restoreRgbPreview() {
    var item = $$(".carousel-item")[live.idx];
    var img = $("#stage-img");
    if (img && item) img.src = item.dataset.src;
  }

  function iceRefresh() { if (iceViewActive()) renderIceView(); else restoreRgbPreview(); }

  function wireIce() {
    var panel = $("#ice-panel");
    if (!panel) return;
    if (live.rollId == null) live.rollId = panel.dataset.rollId;
    var onT = $("#ice-on-toggle"), mT = $("#ice-mask-toggle");
    if (onT) onT.addEventListener("click", function () {
      ice.on = !ice.on; onT.classList.toggle("is-on", ice.on);
      onT.setAttribute("aria-pressed", ice.on); iceRefresh();
    });
    if (mT) mT.addEventListener("click", function () {
      ice.mask = !ice.mask; mT.classList.toggle("is-on", ice.mask);
      mT.setAttribute("aria-pressed", ice.mask); iceRefresh();
    });
    [["#ice-thresh", "ir_thresh"], ["#ice-min-size", "ir_min_size"], ["#ice-kernel", "ir_kernel"]].forEach(function (pair) {
      var inp = $(pair[0]);
      if (!inp) return;
      inp.addEventListener("input", function () {
        var v = panel.querySelector('.grade-val[data-for="' + pair[1] + '"]');
        if (v) v.textContent = String(parseFloat(inp.value));
        if (ice.mask) scheduleIceView();        // detection sliders only change the mask overlay
      });
    });
  }

  function applyGradeToForm(g) {
    // Push a grade object onto the slider inputs (+ readouts + HSL proxies).
    var form = $("#grade-form");
    if (!form || !g) return;
    $$('input[name]', form).forEach(function (inp) {
      if (inp.name in g) {
        inp.value = g[inp.name];
        var v = form.querySelector('.grade-val[data-for="' + inp.name + '"]');
        if (v) v.textContent = String(parseFloat(g[inp.name]));
      }
    });
    if (form._hslReload) form._hslReload();        // refresh HSL proxies for the active band
  }

  function loadFrameGrade(idx) {
    // Pull the selected frame's stored grade into the sliders.
    var item = $$(".carousel-item")[idx];
    if (!item || !item.dataset.grade) return;
    try { applyGradeToForm(JSON.parse(item.dataset.grade)); } catch (e) { return; }
    live.dirty = false;
  }

  function currentGrade() {
    // All NAMED inputs (master + region sliders + the hidden HSL band values).
    // The HSL proxy sliders carry data-hsl and no name, so they're excluded.
    var form = $("#grade-form");
    var g = {};
    if (form) $$('input[name]', form).forEach(function (i) { g[i.name] = parseFloat(i.value); });
    return g;
  }

  function refreshFrame(idx) {
    // After a per-frame Apply: swap the committed JPEG back in (cache-busted),
    // remember the new grade on the item, no full reload (keeps the selection).
    var item = $$(".carousel-item")[idx];
    if (!item) return;
    var src = item.dataset.src + "?t=" + Date.now();
    var thumb = item.querySelector("img");
    if (thumb) thumb.src = src;
    if (live.idx === idx) { var s = $("#stage-img"); if (s) s.src = src; }
    item.dataset.grade = JSON.stringify(currentGrade());
    live.dirty = false;
  }

  // -------- roll detail: frame carousel -------------------------------

  function wireRollCarousel() {
    var carousel = $("#roll-carousel");
    var img = $("#stage-img");
    if (!carousel || !img) return;
    var items = $$(".carousel-item", carousel);
    if (!items.length) return;
    var caption = $("#stage-caption");
    var total = items.length;

    function select(idx, scroll) {
      if (idx < 0) idx = 0;
      if (idx > total - 1) idx = total - 1;
      var item = items[idx];
      if (!item) return;
      items.forEach(function (el) { el.classList.toggle("is-active", el === item); });
      img.src = item.dataset.src;
      img.alt = "Frame " + item.dataset.label;
      if (caption) caption.textContent = "Frame " + item.dataset.label + " / " + total;
      live.idx = idx;
      loadFrameGrade(idx);                       // sliders follow the selected frame
      if (iceViewActive()) scheduleIceView();    // keep the ICE view on the new frame
      if (scroll !== false) {
        item.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
      }
    }

    items.forEach(function (item, i) {
      item.addEventListener("click", function () { select(i); });
    });

    document.addEventListener("keydown", function (ev) {
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
      var active = carousel.querySelector(".carousel-item.is-active");
      var cur = active ? items.indexOf(active) : 0;
      ev.preventDefault();
      select(cur + (ev.key === "ArrowRight" ? 1 : -1));
    });
  }

  // -------- global action delegation ----------------------------------

  document.addEventListener("click", function (ev) {
    var el = ev.target.closest("[data-action]");
    if (!el) return;
    var action = el.dataset.action;
    if (action === "connect") { ev.preventDefault(); connectScanner(); }
    else if (action === "scan-roll") {
      ev.preventDefault();
      var ir = $("#ir-toggle") && $("#ir-toggle").checked;
      scanIntoRoll(el.dataset.rollId, ir);
    }
    else if (action === "process-roll") {
      ev.preventDefault();
      jsonFetch("/api/rolls/" + el.dataset.rollId + "/process", { method: "POST" })
        .then(function () { toast("Generating previews…"); })
        .catch(function (err) { toast("Could not start: " + err.message, "err"); });
    }
    else if (action === "apply-grade") {
      ev.preventDefault();
      if (!$("#grade-form")) return;
      var grade = currentGrade();
      jsonFetch("/api/rolls/" + el.dataset.rollId + "/frame/" + live.idx + "/grade",
                { method: "POST", body: grade })
        .then(function () { toast("Applying grade to frame…"); })
        .catch(function (err) { toast("Grade failed: " + err.message, "err"); });
    }
    else if (action === "export-frame") {
      ev.preventDefault();
      toast("Exporting full-res image…");
      jsonFetch("/api/rolls/" + el.dataset.rollId + "/frame/" + live.idx + "/export",
                { method: "POST", body: currentGrade() })
        .then(function (d) { toast("Exported " + d.exported + " → " + d.dir + "/", "ok"); })
        .catch(function (err) { toast("Export failed: " + err.message, "err"); });
    }
    else if (action === "rotate-left" || action === "rotate-right") {
      ev.preventDefault();
      var gf = $("#grade-form"); if (!gf) return;
      var rin = gf.querySelector('input[name="rotate"]'); if (!rin) return;
      var cur = parseInt(rin.value, 10) || 0;
      rin.value = ((cur + (action === "rotate-right" ? 90 : -90)) % 360 + 360) % 360;
      // Commit immediately: re-render + persist this frame at the new orientation.
      jsonFetch("/api/rolls/" + el.dataset.rollId + "/frame/" + live.idx + "/grade",
                { method: "POST", body: currentGrade() })
        .then(function () { toast("Rotating…"); })
        .catch(function (err) { toast("Rotate failed: " + err.message, "err"); });
      if (typeof iceViewActive === "function" && iceViewActive()) scheduleIceView();
    }
    else if (action === "copy-grade") {
      ev.preventDefault();
      if (!$("#grade-form")) return;
      var cg = currentGrade(); delete cg.rotate;       // copy colour, not orientation
      try { localStorage.setItem("psix_grade_clipboard", JSON.stringify(cg)); } catch (e) {}
      toast("Grade copied", "ok");
    }
    else if (action === "reset-grade") {
      ev.preventDefault();
      var rf = $("#grade-form");
      if (!rf || !rf.dataset.defaults) return;
      var dg; try { dg = JSON.parse(rf.dataset.defaults); } catch (e) { return; }
      var rr = rf.querySelector('input[name="rotate"]');
      if (rr) dg = Object.assign({}, dg, { rotate: parseInt(rr.value, 10) || 0 });   // keep orientation
      applyGradeToForm(dg);
      live.dirty = true;
      if (typeof iceViewActive === "function" && iceViewActive()) renderIceView(); else scheduleLive();
      toast("Grade reset — Apply to commit", "ok");
    }
    else if (action === "paste-grade") {
      ev.preventDefault();
      if (!$("#grade-form")) return;
      var raw = null;
      try { raw = localStorage.getItem("psix_grade_clipboard"); } catch (e) {}
      if (!raw) { toast("No copied grade yet", "warn"); return; }
      var g; try { g = JSON.parse(raw); } catch (e) { toast("Clipboard unreadable", "err"); return; }
      applyGradeToForm(g);
      live.dirty = true;
      if (typeof iceViewActive === "function" && iceViewActive()) renderIceView(); else scheduleLive();
      jsonFetch("/api/rolls/" + el.dataset.rollId + "/frame/" + live.idx + "/grade",
                { method: "POST", body: currentGrade() })
        .then(function () { toast("Grade pasted to frame " + (live.idx + 1), "ok"); })
        .catch(function (err) { toast("Paste failed: " + err.message, "err"); });
    }
    else if (action === "apply-ice") {
      ev.preventDefault();
      jsonFetch("/api/rolls/" + el.dataset.rollId + "/apply_ice",
                { method: "POST", body: iceParams() })
        .then(function () { toast("Applying ICE — re-developing negatives…"); })
        .catch(function (err) { toast("Apply ICE failed: " + err.message, "err"); });
    }
  });

  function wireGradeForm() {
    var form = $("#grade-form");
    if (!form) return;
    live.rollId = form.dataset.rollId;
    // Master / Shadows / Mids / Highlights tabs toggle which slider group shows.
    $$(".grade-tab", form).forEach(function (tab) {
      tab.addEventListener("click", function () {
        $$(".grade-tab", form).forEach(function (t) { t.classList.toggle("is-active", t === tab); });
        $$(".grade-group", form).forEach(function (g) { g.hidden = g.dataset.group !== tab.dataset.tab; });
      });
    });
    $$('input[type="range"]', form).forEach(function (inp) {
      if (inp.dataset.hsl) return;             // HSL proxy sliders handled by wireHsl
      inp.addEventListener("input", function () {
        var v = form.querySelector('.grade-val[data-for="' + inp.name + '"]');
        if (v) v.textContent = String(parseFloat(inp.value));
        live.dirty = true;
        scheduleLive();                        // live-update the centre frame
      });
    });
    wireHsl(form);
  }

  function wireHsl(form) {
    var bandBtns = $$(".hsl-band", form);
    if (!bandBtns.length) return;
    var prox = {};
    $$('input[data-hsl]', form).forEach(function (i) { prox[i.dataset.hsl] = i; });
    var active = "red";
    function hidden(band, ax) { return form.querySelector('input[name="' + band + "_" + ax + '"]'); }
    function readout(ax) {
      var s = form.querySelector('.grade-val[data-hsl="' + ax + '"]');
      if (s) s.textContent = String(parseFloat(prox[ax].value));
    }
    function hsl(deg, l) { return "hsl(" + (((deg % 360) + 360) % 360) + ",70%," + l + "%)"; }
    function paintTracks(center) {
      // Hue track = the rotation range (left = lower hue, right = higher hue).
      prox.hue.style.background = "linear-gradient(90deg," +
        hsl(center - 40, 50) + "," + hsl(center, 50) + "," + hsl(center + 40, 50) + ")";
      // Saturation: grey (left/less) -> full band colour (right/more).
      prox.sat.style.background = "linear-gradient(90deg,#6b7280," + hsl(center, 50) + ")";
      // Luminance: dark -> light, in the band's hue.
      prox.lum.style.background = "linear-gradient(90deg," +
        hsl(center, 18) + "," + hsl(center, 50) + "," + hsl(center, 82) + ")";
    }
    function loadBand(b) {
      active = b;
      var btn = null;
      bandBtns.forEach(function (x) {
        var on = x.dataset.band === b;
        x.classList.toggle("is-active", on);
        if (on) btn = x;
      });
      ["hue", "sat", "lum"].forEach(function (ax) {
        var hv = hidden(b, ax);
        prox[ax].value = hv ? hv.value : 0;
        readout(ax);
      });
      if (btn) paintTracks(parseFloat(btn.dataset.hue));
    }
    bandBtns.forEach(function (btn) {
      btn.addEventListener("click", function () { loadBand(btn.dataset.band); });
    });
    ["hue", "sat", "lum"].forEach(function (ax) {
      prox[ax].addEventListener("input", function () {
        var hv = hidden(active, ax);
        if (hv) hv.value = prox[ax].value;     // write the active band's real value
        readout(ax);
        live.dirty = true;
        scheduleLive();
      });
    });
    loadBand(active);
    form._hslReload = function () { loadBand(active); };
  }

  // -------- firmware first-run upload ----------------------------------
  // Delegated handlers: #scanner-info is replaced wholesale on every device
  // event, so listeners must live on document, not on the card itself.

  async function uploadFirmware(fileList) {
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) return;
    var fd = new FormData();
    files.forEach(function (f) { fd.append("firmware", f); });
    toast("Uploading firmware…");
    try {
      var r = await fetch("/api/firmware/upload", { method: "POST", body: fd });
      var d = await r.json();
      if ((!d.saved || !d.saved.length)) {
        throw new Error((d.errors && d.errors[0] && d.errors[0].error) || d.error || "upload failed");
      }
      if (d.errors && d.errors.length) {
        d.errors.forEach(function (e) { toast(e.file + ": " + e.error, "warn"); });
      }
      toast("Firmware added — loading scanner…", "ok");
      var info = $("#scanner-info");
      if (info) { try { info.innerHTML = await htmlFetch("/partials/scanner-info"); } catch (e) {} }
    } catch (err) {
      toast(String(err.message || err), "err");
    }
  }

  function wireFirmware() {
    document.addEventListener("click", function (ev) {
      if (ev.target.closest && ev.target.closest('[data-action="fw-pick"]')) {
        var inp = $("#fw-file"); if (inp) inp.click();
      }
    });
    document.addEventListener("change", function (ev) {
      if (ev.target && ev.target.id === "fw-file") uploadFirmware(ev.target.files);
    });
    function zone(ev) { return ev.target.closest && ev.target.closest('[data-action="fw-drop"]'); }
    ["dragenter", "dragover"].forEach(function (t) {
      document.addEventListener(t, function (ev) {
        var z = zone(ev); if (z) { ev.preventDefault(); z.classList.add("fw-drop--over"); }
      });
    });
    document.addEventListener("dragleave", function (ev) {
      var z = zone(ev); if (z && !z.contains(ev.relatedTarget)) z.classList.remove("fw-drop--over");
    });
    document.addEventListener("drop", function (ev) {
      var z = zone(ev);
      if (z) {
        ev.preventDefault();
        z.classList.remove("fw-drop--over");
        if (ev.dataTransfer && ev.dataTransfer.files) uploadFirmware(ev.dataTransfer.files);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    wireConfigForms();
    wireScanPage();
    wireRollCarousel();
    wireGradeForm();
    wireIce();
    wireFirmware();
    connectEventStream();
  });
})();
