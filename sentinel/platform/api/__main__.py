# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""``python -m sentinel.platform.api [--host 127.0.0.1] [--port 8000]``"""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentinel.platform.api", description="SENTINEL Console API")
    ap.add_argument("--host", default=os.environ.get("SENTINEL_API_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SENTINEL_API_PORT", "8000")))
    ap.add_argument("--log-level", default="info")
    a = ap.parse_args(argv)
    import uvicorn

    uvicorn.run(
        "sentinel.platform.api.app:create_app", factory=True, host=a.host, port=a.port, log_level=a.log_level
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
