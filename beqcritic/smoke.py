from __future__ import annotations

import json
import platform
import sys


def _try_version(modname: str) -> str | None:
    try:
        mod = __import__(modname)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def main() -> None:
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "beqcritic": _try_version("beqcritic"),
        "torch": _try_version("torch"),
        "transformers": _try_version("transformers"),
        "datasets": _try_version("datasets"),
        "accelerate": _try_version("accelerate"),
    }
    try:
        import torch

        info.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

