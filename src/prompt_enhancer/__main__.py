"""Run the local API with ``python -m prompt_enhancer``."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "prompt_enhancer.api:app",
        host=os.getenv("PROMPT_ENHANCER_HOST", "127.0.0.1"),
        port=int(os.getenv("PROMPT_ENHANCER_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
