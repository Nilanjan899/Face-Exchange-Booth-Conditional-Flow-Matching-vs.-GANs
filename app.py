import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from gan_inference import load_GAN_pipeline
from flow_inference import load_flow_pipeline

# ---- CONFIG: edit these paths ------------------------------------------
GAN_CKPT = r"deepfake_GAN\ckpt_final_step205615_GAN_final.zip"
FLOW_CKPT = r"deepfake_conditional_flow\best.zip"
BISENET_CKPT = r"deepfake_conditional_flow\79999_iter.pth"
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR = Path("uploads")
# --------------------------------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB per upload request

print("Loading GAN...")
gan = load_GAN_pipeline(checkpoint=GAN_CKPT, bisenet_checkpoint=BISENET_CKPT)
print("Loading Flow...")
flow = load_flow_pipeline(checkpoint=FLOW_CKPT, bisenet_checkpoint=BISENET_CKPT)
locks = {"gan": threading.Lock(), "flow": threading.Lock()}


def image_names():
    return sorted(p.name for p in IMAGES_DIR.iterdir() if p.suffix.lower() in EXTS)


def upload_names():
    return sorted(
        "uploads/" + p.name for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in EXTS
    )


def resolve(name):
    """Map a client-side image name to a real path, or None if it is unknown."""
    if name in image_names():
        return str(IMAGES_DIR / name)
    if name in upload_names():
        return str(UPLOAD_DIR / name.split("/", 1)[1])
    return None


def run_gan(src, tgt):
    out = gan(source=src, target=tgt)["image"]
    if out.ndim == 4:
        out = out[0]
    return to_pil_image(out.cpu().clamp(0, 1))


def run_flow(src, tgt):
    return flow(source=src, target=tgt, seed=42)["image"]


RUNNERS = {"gan": run_gan, "flow": run_flow}


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/images")
def list_images():
    return jsonify(image_names())


@app.get("/images/<path:name>")
def serve_image(name):
    return send_from_directory(IMAGES_DIR, name)


@app.get("/uploads/<path:name>")
def serve_upload(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.get("/api/uploads")
def list_uploads():
    return jsonify(upload_names())


@app.post("/api/upload")
def upload():
    saved = []
    for f in request.files.getlist("files"):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in EXTS:
            return jsonify(error=f"{f.filename}: use JPG, PNG or WEBP."), 400
        try:
            Image.open(f.stream).verify()  # confirm it is really an image
            f.stream.seek(0)
        except Exception:
            return jsonify(error=f"{f.filename} is not a valid image."), 400
        fname = f"{uuid.uuid4().hex[:10]}{ext}"
        f.save(UPLOAD_DIR / fname)
        saved.append("uploads/" + fname)
    if not saved:
        return jsonify(error="No files received."), 400
    return jsonify(saved)


@app.get("/outputs/<path:name>")
def serve_output(name):
    return send_from_directory(OUTPUT_DIR, name)


@app.post("/api/generate/<model>")
def generate(model):
    if model not in RUNNERS:
        abort(404)
    data = request.get_json(force=True)
    src = resolve(data.get("source"))
    tgt = resolve(data.get("target"))
    if not src or not tgt:
        return jsonify(error="Unknown image selected."), 400
    try:
        with locks[model]:  # one job per model at a time
            img = RUNNERS[model](src, tgt)
        fname = f"{model}_{uuid.uuid4().hex[:8]}.png"
        img.save(OUTPUT_DIR / fname)
        return jsonify(url=f"/outputs/{fname}")
    except Exception as e:  # surface model errors to the UI
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)