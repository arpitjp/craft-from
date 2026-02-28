"""
Colab Script: Image → 3D Mesh (GPU)

Run this in Google Colab with a T4/A100 runtime.
It generates a 3D mesh from an image using TripoSR locally on the GPU,
then downloads the result.

Usage:
  1. Open Google Colab, select GPU runtime
  2. Upload your image or provide a URL
  3. Copy-paste this entire file into a cell and run
  4. Download the output .obj file
"""

# === INSTALL DEPENDENCIES ===
# fmt: off
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "torchaudio",
    "transformers", "trimesh", "Pillow", "rembg", "onnxruntime",
    "huggingface_hub", "accelerate",
])
# fmt: on

# === CONFIG ===
IMAGE_PATH = "input.png"  # change this to your image path or upload to Colab
OUTPUT_PATH = "output_mesh.obj"
REMOVE_BACKGROUND = True
FOREGROUND_RATIO = 0.85
CHUNK_SIZE = 8192

# === PIPELINE ===
import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
assert torch.cuda.is_available(), "GPU required. Change runtime to T4 or A100."

# Download TripoSR model
model_dir = snapshot_download(repo_id="stabilityai/TripoSR", repo_type="model")
sys.path.insert(0, model_dir)

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

device = "cuda:0"
model = TSR.from_pretrained(model_dir, config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(CHUNK_SIZE)
model.to(device)

# Process image
image = Image.open(IMAGE_PATH)
if REMOVE_BACKGROUND:
    image = remove_background(image)
    image = resize_foreground(image, FOREGROUND_RATIO)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255).astype(np.uint8))

# Generate mesh
with torch.no_grad():
    scene_codes = model([image], device=device)

mesh = model.extract_mesh(scene_codes)[0]
mesh.export(OUTPUT_PATH)

print(f"Mesh saved: {OUTPUT_PATH}")
print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

# Download in Colab
try:
    from google.colab import files
    files.download(OUTPUT_PATH)
except ImportError:
    print(f"Not in Colab. File saved to: {os.path.abspath(OUTPUT_PATH)}")
