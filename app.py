"""
LUMIS — Skin Intelligence Backend
Flask API server that receives a base64 selfie and returns
skin analysis scores + skincare tips via Anthropic Claude.

Setup:
    pip install flask flask-cors anthropic python-dotenv

Run:
    python app.py

Endpoints:
    POST /analyse   — Accepts { image_b64, media_type } → returns analysis JSON
    GET  /health    — Health check
"""

import os
import json
import base64
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1200

SYSTEM_PROMPT = """You are an expert AI dermatology assistant with deep knowledge of skin physiology, 
dermatological assessment, and evidence-based skincare. You analyse facial photographs clinically 
and objectively. You never identify individuals — you focus solely on skin characteristics.
Always respond with valid JSON only, no preamble, no markdown fences."""

ANALYSIS_PROMPT = """Analyse this photograph. First, count the number of clearly visible human faces.

STEP 1 — Face check. Set "face_status" based on what you see:
- "ok"       → exactly one human face, clearly visible and close enough to analyse skin
- "none"     → no human face detected
- "multiple" → more than one human face detected
- "too_far"  → one face present but too small/distant for accurate skin analysis (face covers less than ~30% of frame)

STEP 2 — If face_status is NOT "ok", return ONLY this minimal JSON and nothing else:
{
  "face_status": "<none|multiple|too_far>"
}

STEP 3 — If face_status is "ok", return the full analysis JSON:
{
  "face_status": "ok",
  "overall": <float 1.0–10.0, one decimal place>,
  "verdict": "<one of: Excellent | Very Good | Good | Fair | Needs Attention>",
  "tagline": "<a poetic 5–8 word insight about their skin>",
  "params": [
    { "name": "Acne & Blemishes", "score": <float 1.0–10.0>, "note": "<precise 12-word clinical observation>" },
    { "name": "Hydration",        "score": <float 1.0–10.0>, "note": "<precise 12-word clinical observation>" },
    { "name": "Redness & Tone",   "score": <float 1.0–10.0>, "note": "<precise 12-word clinical observation>" },
    { "name": "Texture & Pores",  "score": <float 1.0–10.0>, "note": "<precise 12-word clinical observation>" },
    { "name": "Dark Spots",       "score": <float 1.0–10.0>, "note": "<precise 12-word clinical observation>" }
  ],
  "tips": [
    { "title": "<3–6 word tip name>", "body": "<2 actionable sentences>", "priority": "high" },
    { "title": "<3–6 word tip name>", "body": "<2 actionable sentences>", "priority": "high" },
    { "title": "<3–6 word tip name>", "body": "<2 actionable sentences>", "priority": "medium" },
    { "title": "<3–6 word tip name>", "body": "<2 actionable sentences>", "priority": "low" }
  ]
}

Scoring guide: 9–10 = flawless, 7–8.9 = good, 5–6.9 = moderate, 3–4.9 = noticeable, 1–2.9 = severe.
Rules: Be realistic and clinically grounded. Do NOT identify the person. Return ONLY the JSON object."""


def validate_image(image_b64: str, media_type: str) -> tuple[bool, str]:
    """Basic validation of incoming image data."""
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if media_type not in allowed_types:
        return False, f"Unsupported media type: {media_type}"
    if not image_b64:
        return False, "Missing image data"
    # Rough size check — base64 of a ~8MB image
    if len(image_b64) > 11_000_000:
        return False, "Image too large (max ~8MB)"
    try:
        base64.b64decode(image_b64, validate=True)
    except Exception:
        return False, "Invalid base64 encoding"
    return True, ""


def call_claude(image_b64: str, media_type: str) -> dict:
    """Send image to Claude and parse the skin analysis response."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": ANALYSIS_PROMPT,
                    },
                ],
            }
        ],
    )

    raw_text = "".join(
        block.text for block in message.content if hasattr(block, "text")
    )
    log.info("Claude response received (%d chars), input_tokens=%d, output_tokens=%d",
             len(raw_text), message.usage.input_tokens, message.usage.output_tokens)

    # Strip any accidental markdown fences and extract JSON
    clean = raw_text.strip()
    if "```" in clean:
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                clean = part
                break
    # Find the first { and last } to extract just the JSON object
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1:
        clean = clean[start:end+1]

    return json.loads(clean)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL}), 200


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Expects JSON body:
        {
            "image_b64":  "<base64-encoded image bytes>",
            "media_type": "image/jpeg"   (or image/png / image/webp)
        }

    Returns JSON:
        {
            "ok": true,
            "result": { overall, verdict, tagline, params, tips }
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"ok": False, "error": "Request body must be JSON"}), 400

    image_b64 = body.get("image_b64", "")
    media_type = body.get("media_type", "image/jpeg")

    valid, err = validate_image(image_b64, media_type)
    if not valid:
        return jsonify({"ok": False, "error": err}), 400

    try:
        result = call_claude(image_b64, media_type)
    except json.JSONDecodeError as e:
        log.error("Failed to parse Claude response as JSON: %s", e)
        return jsonify({"ok": False, "error": "Model returned malformed JSON"}), 502
    except anthropic.AuthenticationError:
        log.error("Invalid Anthropic API key")
        return jsonify({"ok": False, "error": "Invalid API key — check ANTHROPIC_API_KEY"}), 401
    except anthropic.RateLimitError:
        log.warning("Anthropic rate limit hit")
        return jsonify({"ok": False, "error": "Rate limit reached, please retry shortly"}), 429
    except anthropic.APIError as e:
        log.error("Anthropic API error: %s", e)
        return jsonify({"ok": False, "error": f"Anthropic API error: {str(e)}"}), 502
    except Exception as e:
        log.exception("Unexpected error during analysis")
        return jsonify({"ok": False, "error": "Internal server error"}), 500

    # Handle face validation failures
    face_status = result.get("face_status", "ok")
    if face_status != "ok":
        log.info("Face check failed: %s", face_status)
        return jsonify({"ok": False, "face_status": face_status}), 200

    return jsonify({"ok": True, "result": result}), 200


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("⚠️  ANTHROPIC_API_KEY not set — requests will fail at the API call step")
    else:
        log.info("✓ ANTHROPIC_API_KEY loaded")

    port = int(os.environ.get("PORT", 5000))
    log.info("Starting LUMIS backend on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
