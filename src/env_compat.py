"""
env_compat.py — import before `transformers` on machines with a broken torchvision.

`transformers` imports torchvision opportunistically (for image/video models).
If the installed torchvision was built for a different torch version, that import
crashes even for text-only models. Hiding it makes transformers treat it as
absent; none of the text models used here (LLM, DeBERTa NLI, MiniLM) need it.
"""

import os
import sys
import warnings

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")


def _torchvision_ok() -> bool:
    try:
        import torchvision  # noqa: F401
        return True
    except Exception:
        sys.modules.pop("torchvision", None)
        return False


if "torchvision" not in sys.modules and not _torchvision_ok():
    sys.modules["torchvision"] = None

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
