import cv2
import numpy as np
import torch
from pathlib import Path

# Simple wrapper that uses LivePortrait "humans" mode.
# You can later swap to FasterLivePortrait TensorRT paths (see comments below).

class LivePortraitNode:
    def __init__(self, device="cuda", size=512, backend="torch"):
        self.device = device
        self.size = size
        self.backend = backend

        # ---- Plain LivePortrait (torch) ----
        # The official repo ships an inference pipeline via inference.py/gradio_pipeline.
        # We'll import the modules directly to keep it lightweight.
        import sys
        sys.path.append("/app/LivePortrait")
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.gradio_pipeline import GradioPipeline

        self.InferenceConfig = InferenceConfig
        self.CropConfig = CropConfig
        self.GradioPipeline = GradioPipeline

        # Create pipeline
        self.pipe = self.GradioPipeline(
            inference_config=self.InferenceConfig(), crop_config=self.CropConfig()
        )
        # Warmup on first call

    def _prep(self, img_np):
        # center-crop to square, then resize to self.size
        h, w = img_np.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        sq = img_np[y0:y0+side, x0:x0+side]
        if self.size != side:
            sq = cv2.resize(sq, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)  # LivePortrait expects RGB

    def animate(self, source_bgr, driver_bgr):
        # Returns RGB HxWx3 (uint8)
        src = self._prep(source_bgr)
        drv = self._prep(driver_bgr)

        # The gradio pipeline exposes a 'run' like interface taking numpy images.
        # It returns a PIL image / numpy frame.
        out = self.pipe.infer_single_frame(source_img=src, driving_img=drv)
        # Ensure numpy uint8 RGB
        if hasattr(out, 'numpy'):
            out = out.numpy()
        out = np.asarray(out)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out  # RGB
