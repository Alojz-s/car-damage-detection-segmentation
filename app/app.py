"""Gradio demo: upload a photo, see damage masks and the DSI."""

from __future__ import annotations

import json
from pathlib import Path

I18N_DIR = Path(__file__).parent / "i18n"


def load_strings(lang: str = "pl") -> dict[str, str]:
    return json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
