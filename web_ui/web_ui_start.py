import argparse
import os

from web_ui.app import launch_app


def ensure_localhost_bypass_proxy():
    bypass_items = ["127.0.0.1", "localhost", "::1"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        for item in bypass_items:
            if item not in parts:
                parts.append(item)
        os.environ[key] = ",".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "2345")))
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    # Host is fixed to 127.0.0.1
    host = "127.0.0.1"

    ensure_localhost_bypass_proxy()

    launch_app(
        server_name=host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
