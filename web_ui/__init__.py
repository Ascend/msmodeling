"""Web UI for msmodeling simulation workflows."""


def launch_app(*args, **kwargs):
    from .app import launch_app as _launch_app

    return _launch_app(*args, **kwargs)


__all__ = ["launch_app"]
