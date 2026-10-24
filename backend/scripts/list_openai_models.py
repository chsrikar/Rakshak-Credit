"""List model IDs visible to the configured OpenAI API key without printing secrets."""

from pathlib import Path
import os
import sys

import httpx
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env", override=False)


def main() -> int:
    api_key = os.getenv("LUNA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing LUNA_API_KEY (or OPENAI_API_KEY); no API request was made.", file=sys.stderr)
        return 2

    try:
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(15.0, connect=5.0),
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        model_ids = sorted({item["id"] for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)})
        if not model_ids:
            print("The API returned no model IDs.", file=sys.stderr)
            return 1
        print("Model IDs available to this API key:")
        print("\n".join(model_ids))
        configured = os.getenv("LUNA_MODEL_NAME")
        if configured:
            print(f"Configured LUNA_MODEL_NAME is listed: {configured in model_ids}")
        return 0
    except httpx.HTTPStatusError as exc:
        print(f"OpenAI model discovery failed with HTTP {exc.response.status_code}.", file=sys.stderr)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        print(f"OpenAI model discovery failed ({type(exc).__name__}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
