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

ANALYSIS_PROMPT = """Analyse this facial photograph and provide a comprehensive skin health assessment.

Return ONLY a valid JSON object with this exact schema:
{
  "overall": <float 1.0–10.0, one decimal place>,
  "verdict": "<one of: Excellent | Very Good | Good | Fair | Needs Attention>",
  "tagline": "<a poetic 5–8 word insight about their skin>",
  "params": [
    {
      "name": "Acne & Blemishes",
      "score": <float 1.0–10.0>,
      "note": "<precise 12-word clinical observation about acne, breakouts, congestion>"
    },
    {
      "name": "Hydration",
      "score": <float 1.0–10.0>,
      "note": "<precise 12-word clinical observation about moisture, plumpness, dryness>"
    },
    {
      "name": "Redness & Tone",
      "score": <float 1.0–10.0>,
      "note": "<precise 12-word clinical observation about tone evenness, redness, flushing>"
    },
    {
      "name": "Texture & Pores",
      "score": <float 1.0–10.0>,
      "note": "<precise 12-word clinical observation about surface smoothness and pore visibility>"
    },
    {
      "name": "Dark Spots",
      "score": <float 1.0–10.0>,
      "note": "<precise 12-word clinical observation about pigmentation, sun damage, marks>"
    }
  ],
  "tips": [
    {
      "title": "<concise elegant tip name, 3–6 words>",
      "body": "<2 specific, actionable sentences with product types or ingredients>",
      "priority": "high"
    },
    {
      "title": "<concise elegant tip name, 3–6 words>",
      "body": "<2 specific, actionable sentences with product types or ingredients>",
      "priority": "high"
    },
    {
      "title": "<concise elegant tip name, 3–6 words>",
      "body": "<2 specific, actionable sentences with product types or ingredients>",
      "priority": "medium"
    },
    {
      "title": "<concise elegant tip name, 3–6 words>",
      "body": "<2 specific, actionable sentences with product types or ingredients>",
      "priority": "low"
    }
  ]
}

Scoring guide:
- 9.0–10.0 = Flawless / near-perfect
- 7.0–8.9  = Good, minor concerns
- 5.0–6.9  = Moderate concerns
- 3.0–4.9  = Noticeable issues
- 1.0–2.9  = Severe concern

Rules:
- Be realistic, nuanced, and clinically grounded
- Tips must directly address the lowest-scoring parameters first (high priority)
- If the image does not clearly show a face, return overall=5.0 with useful generic advice
- Do NOT identify the person — analyse skin characteristics only
- Return ONLY the JSON object, nothing else"""


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

    # Strip any accidental markdown fences
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip().rstrip("`").strip()

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
