"""
train_model.py — Train YOLOv8-Seg on the crack-seg dataset

Trains a YOLOv8 nano segmentation model on the Ultralytics built-in
crack-seg dataset (~4,029 images). Optimised for Apple Silicon M4
with MPS acceleration.

Usage:
    python train_model.py
"""

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "yolov8n-seg.pt"       # Pretrained nano segmentation model
DATASET    = "crack-seg.yaml"       # Built-in Ultralytics dataset (auto-downloads)
EPOCHS     = 100
PATIENCE   = 30                     # Early stopping — stops if no improvement
BATCH      = 8                      # Safe for 16 GB unified memory
IMGSZ      = 640
CACHE      = True                   # Cache images in RAM for faster I/O
PROJECT    = "runs/segment"         # Training output directory
NAME       = "crack_seg"            # Run name
OUTPUT_DIR = Path("models")         # Where to copy the best weights


def get_device() -> str:
    """Pick the best available device: MPS → CPU."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ Using Apple Silicon GPU (MPS)")
            return "mps"
    except ImportError:
        pass
    print("⚠️  MPS unavailable — falling back to CPU (this will be much slower)")
    return "cpu"


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    device = get_device()

    print(f"""
╔══════════════════════════════════════════════════╗
║   Bridge Crack Detection — Model Training        ║
╠══════════════════════════════════════════════════╣
║  Base model  :  {BASE_MODEL:<30s}  ║
║  Dataset     :  {DATASET:<30s}  ║
║  Epochs      :  {EPOCHS:<30d}  ║
║  Patience    :  {PATIENCE:<30d}  ║
║  Batch size  :  {BATCH:<30d}  ║
║  Image size  :  {IMGSZ:<30d}  ║
║  Device      :  {device:<30s}  ║
║  Cache       :  {str(CACHE):<30s}  ║
╚══════════════════════════════════════════════════╝
""")

    # ------------------------------------------------------------------
    # 1. Load pretrained model
    # ------------------------------------------------------------------
    print("📦 Loading pretrained model...")
    model = YOLO(BASE_MODEL)

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    print("🚀 Starting training — this will take a while...\n")
    results = model.train(
        data=DATASET,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        imgsz=IMGSZ,
        device=device,
        cache=CACHE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        plots=True,         # Generate training plots
        save=True,           # Save checkpoints
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 3. Copy best weights to models/
    # ------------------------------------------------------------------
    best_weights = Path(PROJECT) / NAME / "weights" / "best.pt"
    if best_weights.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUTPUT_DIR / "crack_seg_best.pt"
        shutil.copy2(best_weights, dest)
        print(f"\n✅ Best weights saved to: {dest}")
    else:
        print(f"\n⚠️  Expected weights at {best_weights} not found.")
        print("   Check the training output directory for results.")
        return

    # ------------------------------------------------------------------
    # 4. Validate on test set and print metrics
    # ------------------------------------------------------------------
    print("\n📊 Running validation on test set...")
    trained_model = YOLO(str(OUTPUT_DIR / "crack_seg_best.pt"))
    metrics = trained_model.val(data=DATASET, split="test", device=device)

    print(f"""
╔══════════════════════════════════════════════════╗
║   Training Complete — Final Metrics              ║
╠══════════════════════════════════════════════════╣
║  Box  mAP50     :  {metrics.box.map50:.4f}                        ║
║  Box  mAP50-95  :  {metrics.box.map:.4f}                        ║
║  Mask mAP50     :  {metrics.seg.map50:.4f}                        ║
║  Mask mAP50-95  :  {metrics.seg.map:.4f}                        ║
╚══════════════════════════════════════════════════╝

Best weights: {OUTPUT_DIR / 'crack_seg_best.pt'}
Training plots: {Path(PROJECT) / NAME}

Next steps:
  1. Restart the backend:  uvicorn main:app --host 0.0.0.0 --port 8000
  2. The API will auto-load the trained model
  3. Upload a crack image via the frontend with use_mock=false
""")


if __name__ == "__main__":
    main()
