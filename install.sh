#!/usr/bin/env bash
#
# psix installer — sets up the local web app, USB permissions, and firmware dir.
# Re-runnable (idempotent). Run from the repo root:
#
#     ./install.sh
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
FWDIR="${PSIX_FIRMWARE_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/psix/firmware}"
RULE_SRC="$ROOT/packaging/99-pakon.rules"
RULE_DST="/etc/udev/rules.d/99-pakon.rules"
BINDIR="$HOME/.local/bin"

# ---- pretty output --------------------------------------------------------
if [ -t 1 ]; then
  B=$(tput bold); R=$(tput sgr0); G=$(tput setaf 2); Y=$(tput setaf 3); C=$(tput setaf 6); E=$(tput setaf 1)
else B=; R=; G=; Y=; C=; E=; fi
step() { printf "\n${B}${C}==>${R} ${B}%s${R}\n" "$1"; }
ok()   { printf "  ${G}✓${R} %s\n" "$1"; }
warn() { printf "  ${Y}!${R} %s\n" "$1"; }
err()  { printf "  ${E}✗${R} %s\n" "$1"; }

# ---- 1. Python ------------------------------------------------------------
step "Checking Python"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then err "Python 3.10+ not found. Install it and re-run."; exit 1; fi
VER=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
if ! "$PY" -c 'import sys;raise SystemExit(0 if sys.version_info[:2]>=(3,10) else 1)'; then
  err "Found Python $VER, but psix needs 3.10+."; exit 1
fi
ok "Python $VER ($PY)"

# ---- 2. venv + install ----------------------------------------------------
step "Installing psix into $VENV"
if [ ! -d "$VENV" ]; then "$PY" -m venv "$VENV" || { err "venv creation failed"; exit 1; }; ok "created venv"; else ok "venv exists"; fi
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null 2>&1
if "$VENV/bin/pip" install "$ROOT" ; then ok "psix + dependencies installed"; else err "pip install failed (see output above)"; exit 1; fi

# ---- 3. launcher on PATH --------------------------------------------------
step "Adding the 'psix' command"
mkdir -p "$BINDIR"
ln -sf "$VENV/bin/psix" "$BINDIR/psix"
ok "linked $BINDIR/psix -> venv"
case ":$PATH:" in
  *":$BINDIR:"*) ok "$BINDIR is on your PATH" ;;
  *) warn "$BINDIR is not on your PATH — add this to your shell profile:"; printf "      export PATH=\"\$HOME/.local/bin:\$PATH\"\n" ;;
esac

# ---- 4. USB permissions (udev) -------------------------------------------
step "USB permissions (udev rule for the scanner)"
if [ ! -f "$RULE_SRC" ]; then
  warn "rule file not found at $RULE_SRC — skipping"
elif cmp -s "$RULE_SRC" "$RULE_DST" 2>/dev/null; then
  ok "udev rule already installed"
else
  warn "psix talks to the scanner over USB in userspace; this needs a udev rule."
  printf "    Install it now with sudo? [Y/n] "; read -r ans </dev/tty 2>/dev/null || ans=n
  if [ -z "$ans" ] || [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
    if sudo cp "$RULE_SRC" "$RULE_DST" && sudo udevadm control --reload-rules && sudo udevadm trigger; then
      ok "udev rule installed — replug the scanner for it to take effect"
    else err "udev install failed; you can run it manually (see README.md)"; fi
  else
    warn "skipped — run 'sudo cp $RULE_SRC $RULE_DST' later, or run psix as root"
  fi
fi

# ---- 5. Firmware dir ------------------------------------------------------
step "Scanner firmware (you supply it — Kodak property, not bundled)"
mkdir -p "$FWDIR"
ok "firmware dir: $FWDIR"
have_fw=$(find "$FWDIR" -maxdepth 1 -iname 'pakon[0-9].hex' 2>/dev/null | head -1)
if [ -n "$have_fw" ]; then
  ok "firmware already present ($(basename "$have_fw"))"
else
  warn "no firmware yet. Put your Pakon firmware files into:"
  printf "      ${B}%s${R}\n" "$FWDIR"
  printf "      • pkninit.hex\n      • pakon5.hex / pakon7.hex / pakon8.hex (psix picks the right one)\n"
  printf "    psix uploads it over USB the first time you connect in Hardware mode.\n"
fi

# ---- done -----------------------------------------------------------------
step "Done"
printf "  Start psix with:  ${B}${G}psix${R}   (it opens http://127.0.0.1:5135 in your browser)\n"
printf "  Then: power on the scanner, switch to ${B}Hardware${R} on the Settings page.\n\n"
