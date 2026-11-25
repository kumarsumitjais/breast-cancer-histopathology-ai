"""Deploy the live demo (frontend + API + model weights) to a Hugging Face Docker Space.

Usage (from the repo root, inside the venv):
    hf auth login                                  # once, with a token that has write access
    python backend/scripts/deploy_space.py [space-name]

Re-run after any code change to redeploy; only changed files are uploaded.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "backend" / "app" / "model" / "densenet121_063A.keras"

# The Space's README front matter configures the Space itself (Docker SDK, port).
SPACE_README = """---
title: Breast Histopathology Classifier
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: DenseNet121 benign vs malignant classifier with Grad-CAM
---

Live demo of a DenseNet121 model that classifies breast histopathology images as benign or malignant,
with Grad-CAM heatmaps.

Source code: https://github.com/kumarsumitjais/breast-cancer-histopathology-ai

**Research use only. Not a medical device.**
"""

name = sys.argv[1] if len(sys.argv) > 1 else "breast-cancer-histopathology-ai"
if not MODEL.exists():
    sys.exit(f"Model weights missing: {MODEL}")

api = HfApi()
repo_id = f"{api.whoami()['name']}/{name}"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=ROOT,
    allow_patterns=["Dockerfile", ".dockerignore", "backend/requirements.txt", "backend/app/**", "frontend/**"],
    ignore_patterns=["**/__pycache__/**", "frontend/node_modules/**", "frontend/dist/**", "**/*.sha256", "**/.env"],
    commit_message="Deploy demo",
)
api.upload_file(
    path_or_fileobj=SPACE_README.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="space",
    commit_message="Space README",
)
print(f"Deployed. Build logs: https://huggingface.co/spaces/{repo_id}")
print(f"Demo URL (once built): https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space")
