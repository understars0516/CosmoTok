import typing

try:
    from typing_extensions import Self as _Self
except Exception:
    _Self = None

if _Self is not None and not hasattr(typing, "Self"):
    setattr(typing, "Self", _Self)

