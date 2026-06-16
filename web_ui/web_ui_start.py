import argparse
import os
import socket

from web_ui.app import launch_app


def get_bind_address() -> str:
    """Auto-detect and return appropriate bind address.

    Prefers IPv4, falls back to IPv6 if unavailable.
    Returns IPv6 addresses in bracket notation required by Gradio.
    """
    # Test IPv4 first
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))  # Port 0: system allocates ephemeral port for testing
        s.close()
        return "127.0.0.1"  # IPv4 available
    except OSError:
        pass

    # IPv4 unavailable, test IPv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.bind(("::1", 0))  # Port 0: system allocates ephemeral port for testing
        s.close()
        return "[::1]"  # IPv6 localhost (Gradio requires brackets)
    except OSError:
        pass

    # Both unavailable, return default
    return "127.0.0.1"


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

    # Auto-detect: prefer IPv4, fallback to IPv6
    host = get_bind_address()
    print(f"Network detected, using address: {host}")

    ensure_localhost_bypass_proxy()

    launch_app(
        server_name=host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
