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
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PORT = 8787
INFERENCE = Path.home() / ".claude/PAI/Tools/Inference.ts"

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
Image 6 = "Spatial leakage" (random vs spatial split toggle, same-farm patches leak)"""

SYSTEM_PROMPT = f"""You are a warm, visual-first ML tutor inside an interactive guide. The student
is a complete beginner building a EuroSAT land-cover classifier (27,000 Sentinel-2 patches, 64x64,
10 classes; ResNet18 ImageNet-pretrained with FROZEN backbone + new 512->10 head; trained 3 epochs
on a MacBook CPU; results: 85.1% validation accuracy, Cohen's kappa 0.834). She is preparing for
ML interviews. {SECTION_MAP}

Answer her question. Respond with ONLY a JSON object, no markdown fences, with EXACTLY these keys:
  "answer": 3-6 beginner-friendly sentences anchored to HER project's real numbers where relevant.
  "tab_title": a 2-4 word title for this concept.
  "is_extension": true if this concept goes beyond what the guide's six sections already cover, else false.
  "visual_html": a COMPLETE self-contained HTML document (it becomes an iframe srcdoc) that VISUALLY
    teaches the answer with an ANIMATION. Rules: inline CSS+JS only, no external resources; dark theme
    (body background #0f1419, text #e8edf2, accent #4fc3f7, orange #ffa726, green #66bb6a); one
    <canvas> about 520x300 animated with requestAnimationFrame; the animation must loop and genuinely
    illustrate the concept (moving parts, labels); a one-line caption under the canvas.
Keep visual_html under 4000 characters. Escape characters correctly so the whole response is valid JSON."""


def ask_model(question: str) -> dict | None:
    """Shell out to the PAI Inference tool; parse its JSON reply."""
    try:
        proc = subprocess.run(
            ["bun", str(INFERENCE), "--level", "standard", "--timeout", "90000",
             SYSTEM_PROMPT, question],
            capture_output=True, text=True, timeout=100)
        raw = proc.stdout.strip()
        # Models sometimes wrap JSON in ```json fences — strip them.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1:
            return None
        data = json.loads(raw[start:end + 1])
        if {"answer", "visual_html", "tab_title"} <= data.keys():
            return data
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
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


class TutorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS), **kwargs)

    def do_POST(self):
        if self.path != "/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            question = json.loads(self.rfile.read(length))["question"][:2000]
        except (json.JSONDecodeError, KeyError):
            self.send_error(400, "expected JSON {question: ...}")
            return
        result = kb_answer(question) or ask_model(question) or apology()
        result.setdefault("source", "ai")
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter logs
        if "/ask" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"AI Tutor serving http://localhost:{PORT}/visual_guide.html")
    ThreadingHTTPServer(("127.0.0.1", PORT), TutorHandler).serve_forever()
