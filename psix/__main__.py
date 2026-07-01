"""psix launcher — `python -m psix` or the `psix` console script.

psix runs as a local web app: it starts a server on 127.0.0.1 and opens the
page in your default browser.  Nothing leaves your machine.

  psix                 start the server and open the browser
  psix --no-browser    start the server only (headless / remote use)

Env: PSIX_HOST, PSIX_PORT, PSIX_NO_BROWSER=1 (same as --no-browser).
"""

import os
import socket
import sys
import threading
import time
import webbrowser

from .app import create_app

HOST = os.environ.get("PSIX_HOST", "127.0.0.1")
PORT = int(os.environ.get("PSIX_PORT", "5135"))


def _wait_and_open(url, timeout=10.0):
    """Open the browser once the server is accepting connections (so the user
    never lands on a connection-refused page)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=0.5):
                break
        except OSError:
            time.sleep(0.15)
    try:
        webbrowser.open(url)
    except Exception:                       # noqa: BLE001 — never let this break the server
        pass


def main():
    app = create_app()
    url = f"http://{HOST}:{PORT}"

    no_browser = (
        "--no-browser" in sys.argv[1:]
        or "--server" in sys.argv[1:]              # back-compat alias
        or os.environ.get("PSIX_NO_BROWSER")
        or os.environ.get("PSIX_SERVER")           # back-compat alias
    )

    print(f"psix: serving at {url}")
    if no_browser:
        print("      (open it in your browser)")
    else:
        print("      opening your browser…  (Ctrl+C here to quit)")
        threading.Thread(target=_wait_and_open, args=(url,), daemon=True).start()

    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()
