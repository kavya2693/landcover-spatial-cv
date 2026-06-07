"""AI Tutor server for the visual guide.

LESSON — why a server at all? A web page (HTML/JS) runs sandboxed in the
browser and cannot launch programs or hold secrets. So the page POSTs the
question to this tiny local server, the server asks Claude (via the PAI
Inference tool, which uses the Claude subscription — no API key), and returns
JSON the page can render. This page→backend→model loop is exactly how every
production AI product works; ours is just 150 lines.

Run:   .venv/bin/python src/tutor_server.py
Open:  http://localhost:8787/visual_guide.html
"""

import json
import re
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PORT = 8787
INFERENCE = Path.home() / ".claude/PAI/Tools/Inference.ts"
INFERENCE_LEVEL = "standard"
INFERENCE_TIMEOUT_S = 220  # CLI latency is high-variance (~2min measured)
# Bound concurrent CLI forks — a burst of questions shouldn't spawn N processes
# all competing for the same Claude subscription.
_inference_slots = threading.Semaphore(2)

sys.path.insert(0, str(ROOT / "src"))
try:
    import tutor_kb  # offline fallback answers (8 prebuilt visual explainers)
except ImportError:
    tutor_kb = None

SECTION_MAP = """The guide's sections the student may reference:
Image 1 = "An image is just a grid of numbers" (hoverable 16x16 pixel grid, RGB values)
Image 2 = "Convolution" (animated 3x3 vertical-edge filter sliding, river edges light up)
Image 3 = "Transfer learning" (frozen blue ResNet18 layers + new orange 512->10 head)
Image 4 = "Training curves" (their real run: loss 0.94->0.47, val acc 82.5%->85.1%, 3 epochs)
Image 5 = "Confusion matrix" (top errors: River->Highway 81x, PermanentCrop->HerbaceousVegetation 49x; kappa 0.834)
Image 6 = "Spatial leakage" (random vs spatial split toggle, same-farm patches leak)
Phase B tab = the real experiment result: random 85.2% vs spatial-50km-blocks 85.6% — an honest
NULL result (no measurable leakage; gap -0.4% is within noise). Hypothesis: the frozen backbone's
tiny 5,130-param head lacks capacity to memorize places; Phase C will test if fine-tuning all 11M
params reopens the gap. Coordinates came from GeoTIFF tags (tiepoint/scale/EPSG) via tifffile."""

CONTEXT = f"""You are a warm ML tutor inside an interactive guide. The student is a beginner
building a EuroSAT land-cover classifier (27,000 Sentinel-2 patches, 64x64, 10 classes; ResNet18
frozen-backbone 85.1% acc / kappa 0.834; full fine-tune 96.4% random / 95.8% spatial; ViT-tiny
benchmark 96.1% spatial). She is preparing for ML interviews. {SECTION_MAP}"""

# Fast path: text only — ~8-15s on the fast model vs ~90s with an animation.
SYSTEM_PROMPT_TEXT = f"""{CONTEXT}

Answer her question. Respond ONLY a JSON object, no markdown fences, EXACTLY these keys:
  "answer": 3-5 beginner-friendly sentences anchored to HER project's real numbers where relevant.
  "tab_title": a 2-4 word title for this concept.
  "is_extension": true if the concept goes beyond the guide's sections, else false.
Total under 800 characters. Valid JSON only."""

# Slow path, on demand only (the "draw it" button): generate the animation.
SYSTEM_PROMPT_VISUAL = f"""{CONTEXT}

The student asked: a question, and got a text answer (both provided). Create ONLY a JSON object,
no markdown fences, with EXACTLY one key:
  "visual_html": a COMPLETE self-contained HTML document (iframe srcdoc) that VISUALLY teaches
    the answer with an ANIMATION. Inline CSS+JS only, no external resources; dark theme (body
    #0f1419, text #e8edf2, accent #4fc3f7, orange #ffa726, green #66bb6a); one
    <canvas width=480 height=240> animated with requestAnimationFrame; loops; 2-4 short labels.
CRITICAL: visual_html under 1600 characters. Terse JS, no comments inside. Valid JSON only."""


def _inference(system_prompt: str, user_prompt: str, level: str,
               timeout_s: int) -> dict | None:
    """Shell out to the PAI Inference tool; parse its JSON reply."""
    try:
        with _inference_slots:
            proc = subprocess.run(
                ["bun", str(INFERENCE), "--level", level,
                 "--timeout", str(timeout_s * 1000),
                 system_prompt, user_prompt],
                capture_output=True, text=True, timeout=timeout_s + 10)
        raw = proc.stdout.strip()
        # Models sometimes wrap JSON in ```json fences — strip them.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1:
            return None
        return json.loads(raw[start:end + 1])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
    return None


def ask_model(question: str) -> dict | None:
    """Fast path: text-only answer on the fast model (~8-15s measured)."""
    data = _inference(SYSTEM_PROMPT_TEXT, question, "fast", 60)
    if data and {"answer", "tab_title"} <= data.keys():
        data["source"] = "ai"
        data["visual_html"] = ""          # visuals are drawn on demand
        data["can_draw"] = True           # tells the UI to offer "draw it"
        return data
    return None


def draw_visual(question: str, answer: str) -> dict | None:
    """Slow path, user-requested: generate the animation for a given answer."""
    prompt = f"Question: {question}\nText answer she received: {answer}"
    data = _inference(SYSTEM_PROMPT_VISUAL, prompt, INFERENCE_LEVEL,
                      INFERENCE_TIMEOUT_S)
    if data and data.get("visual_html"):
        return {"visual_html": data["visual_html"]}
    return None


def kb_answer(question: str) -> dict | None:
    """Curated knowledge base — instant, hand-tuned animations for the 8 most
    common concepts. Checked BEFORE the AI: cheap, fast, highest quality."""
    if tutor_kb:
        hit = tutor_kb.lookup(question)
        if hit:
            return {"answer": hit["answer"], "visual_html": hit["visual_html"],
                    "tab_title": hit["tab_title"], "is_extension": True,
                    "source": "offline-kb"}
    return None


def apology() -> dict:
    return {"answer": "I couldn't reach the AI tutor just now. Try again in a "
                      "moment — or check docs/LEARNING.md and docs/INTERVIEW_QA.md "
                      "which cover most concepts.",
            "visual_html": "", "tab_title": "Unavailable", "is_extension": False,
            "source": "error"}


# Experiment artifacts live in outputs/ (outside docs/); expose just these
RESULT_FILES = {"/spatial_gap.json": "application/json",
                "/spatial_blocks.png": "image/png",
                "/finetune_gap.json": "application/json",
                "/finetune_compare.png": "image/png",
                "/benchmark.json": "application/json",
                "/benchmark_compare.png": "image/png"}


class TutorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS), **kwargs)

    def do_GET(self):
        if self.path in RESULT_FILES:
            path = ROOT / "outputs" / self.path.lstrip("/")
            if not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", RESULT_FILES[self.path])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")  # results update live
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def _send_json(self, result: dict):
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self.send_error(400, "expected JSON")
            return
        if self.path == "/ask":
            question = str(payload.get("question", ""))[:2000]
            if not question:
                self.send_error(400, "expected JSON {question: ...}")
                return
            # force_ai: the "ask Claude instead" escape hatch when a KB card
            # was related-but-not-exactly what she asked
            kb = None if payload.get("force_ai") else kb_answer(question)
            self._send_json(kb or ask_model(question) or apology())
        elif self.path == "/draw":
            result = draw_visual(str(payload.get("question", ""))[:2000],
                                 str(payload.get("answer", ""))[:2000])
            self._send_json(result or {"visual_html": "", "error": "draw failed"})
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):  # quieter logs: only /ask and errors
        if any("/ask" in str(a) for a in args) or "code" in fmt:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"AI Tutor serving http://localhost:{PORT}/visual_guide.html")
    ThreadingHTTPServer(("127.0.0.1", PORT), TutorHandler).serve_forever()
