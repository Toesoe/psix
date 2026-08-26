# psix — Pakon Scanner Interface for Linux

> **Proof of concept.** The goal here was to prove we can drive the scanner from Linux and get real
> image data out of it — firmware load, transport/scan, and a working negative→positive pipeline.
> That works end to end. The UI is far from complete and is not representative of a finished product.

A local app for the **Kodak/Pakon F135/F135+** film scanner on Linux: a userspace USB driver (no kernel
module), the scan/transport control, and a C‑41 negative→positive colour pipeline with live grading
and IR dust/scratch removal (ICE). psix runs as a **local web app** — it starts a server on your
machine and opens it in your browser. Nothing leaves your computer.

```
psix/
  psix/            Flask app + driver
    pakon/         userspace F135+ driver + colour pipeline
    assets/        bundled data (tone curve, EZ-USB loader)
    templates/ static/
  install.sh       one-command setup
```

## Install

```sh
./install.sh
```

This creates a virtualenv, installs psix, adds the `psix` command to `~/.local/bin`, installs the
USB udev rule (asks for sudo), and sets up your firmware folder. Python 3.10+ required.

<details><summary>Manual install</summary>

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install .
```

**USB permissions.** The scanner is accessed via libusb in userspace, so your user needs device
access:
```sh
sudo cp packaging/99-pakon.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
Then replug the scanner. (Without this you'd have to run as root.)
</details>

## Scanner firmware (you supply it — Kodak property, not bundled)

The F135/F135+ boots with no application firmware; psix uploads it over USB. You need **one** file — the
application image for your scanner's hardware revision:

- `pakon5.hex`, `pakon7.hex`, or `pakon8.hex`  (psix reads the revision and picks the right one)

**Easiest:** the first time you connect the scanner in Hardware mode, the Settings page shows a
drop‑zone — just drag your `.hex` file onto it and psix saves it and loads the scanner. Or place it
in the firmware dir yourself: `~/.local/share/psix/firmware/` (or set `$PSIX_FIRMWARE_DIR`).

The generic EZ‑USB second‑stage loader (`ezusb_stage2.ihex`) **is** bundled — that one is not Kodak code.

## Use

```sh
psix                 # starts the server and opens http://127.0.0.1:5135 in your browser
psix --no-browser    # headless / remote: just serve, open the URL yourself
```

1. psix starts in **mock mode** (full UI, no hardware). Switch to **Hardware** on the Settings page.
2. Power on the scanner. psix detects it and — the first time — prompts you to add the firmware (see
   above). It then loads it over USB and initializes the scanner automatically.
3. Scan a roll (3‑channel, or 4‑channel **IR** to enable dust/scratch removal), grade each frame
   live, tune ICE, and export full‑resolution images.

Data lives in your user dir (`~/.local/share/psix` by default; `$PSIX_DATA_DIR` overrides). The
output folder for rolls and raw scans is set on the Settings page.
