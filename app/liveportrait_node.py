# app/liveportrait_node.py
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

class LivePortraitNode:
    """
    Minimal real-time wrapper using LivePortrait core modules.
    - Prepares the source once
    - For each driver frame, computes keypoints & renders one RGB frame
    """
    def __init__(self, device="cuda", size=512, backend="torch"):
        self.device = device
        self.size = size
        self.backend = backend

        # import LivePortrait (mounted at /app/LivePortrait)
        lp_root = "/app/LivePortrait"
        sys.path.append(lp_root)

        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.utils.cropper import Cropper
        from src.utils.camera import get_rotation_matrix
        from src.live_portrait_wrapper import LivePortraitWrapper

        self.InferenceConfig = InferenceConfig
        self.CropConfig = CropConfig
        self.Cropper = Cropper
        self.get_rotation_matrix = get_rotation_matrix
        self.LivePortraitWrapper = LivePortraitWrapper

        # ---- configs for streaming ----
        self.infer_cfg = self.InferenceConfig()
        self.infer_cfg.device_id = 0
        self.infer_cfg.flag_do_crop = True
        self.infer_cfg.flag_write_result = False
        self.infer_cfg.flag_pasteback = False
        self.infer_cfg.flag_stitching = True
        self.infer_cfg.flag_relative = True

        self.crop_cfg = self.CropConfig()

        # ⚠️ IMPORTANT: use positional arg to be compatible with both signatures
        # Some versions: __init__(self, cfg: InferenceConfig)
        # Others:       __init__(self, inference_cfg: InferenceConfig)
        self.wrapper = self.LivePortraitWrapper(self.infer_cfg)
        self.cropper = self.Cropper(crop_cfg=self.crop_cfg, device_id=0)

        # state
        self._reset_state()

    def _reset_state(self):
        self.f_s = None
        self.x_s = None
        self.x_c_s = None
        self.x_s_info = None
        self.R_s = None
        self.R_d_0 = None
        self.x_d_0_info = None

    @staticmethod
    def _center_square_resize_bgr(img_bgr: np.ndarray, size: int) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img_bgr[y0:y0+side, x0:x0+side]
        if crop.shape[0] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        return crop

    def set_source(self, source_bgr: np.ndarray):
        """Prepare source once (crop -> prepare -> kp -> features)."""
        # Crop to face using built-in cropper (expects RGB)
        src_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        crop_info = self.cropper.crop_single_image(src_rgb)  # raises if no face

        # 256x256 tensor (1x3xHxW) for networks
        I_s = self.wrapper.prepare_source(crop_info["img_crop_256x256"])

        # kp info & rotations
        self.x_s_info = self.wrapper.get_kp_info(I_s)
        self.x_c_s = self.x_s_info["kp"]
        self.R_s = self.get_rotation_matrix(
            self.x_s_info["pitch"], self.x_s_info["yaw"], self.x_s_info["roll"]
        )

        # features & transformed kp
        self.f_s = self.wrapper.extract_feature_3d(I_s)
        self.x_s = self.wrapper.transform_keypoint(self.x_s_info)

        # reset driving baseline
        self.R_d_0 = None
        self.x_d_0_info = None

    def animate(self, driver_bgr: np.ndarray) -> np.ndarray:
        """
        One streaming step.
        Returns RGB uint8 frame of shape (H, W, 3).
        """
        assert self.f_s is not None, "Call set_source(...) before animate(...)."

        # Fast square crop + 256 for the driver
        drv_bgr_sq = self._center_square_resize_bgr(driver_bgr, 256)
        drv_rgb_sq = cv2.cvtColor(drv_bgr_sq, cv2.COLOR_BGR2RGB)

        # Build tensor for motion extractor
        I_d = self.wrapper.prepare_source(drv_rgb_sq)  # 1x3x256x256

        # Extract driving kp info & rotation
        x_d_i_info = self.wrapper.get_kp_info(I_d)  # dict of tensors
        R_d_i = self.get_rotation_matrix(
            x_d_i_info["pitch"], x_d_i_info["yaw"], x_d_i_info["roll"]
        )

        # Initialize baseline on first frame
        if self.R_d_0 is None:
            self.R_d_0 = R_d_i
            self.x_d_0_info = x_d_i_info

        # Relative motion (same math as LivePortraitPipeline.execute)
        if self.infer_cfg.flag_relative:
            R_new = (R_d_i @ self.R_d_0.permute(0, 2, 1)) @ self.R_s
            delta_new = self.x_s_info["exp"] + (x_d_i_info["exp"] - self.x_d_0_info["exp"])
            scale_new = self.x_s_info["scale"] * (x_d_i_info["scale"] / self.x_d_0_info["scale"])
            t_new = self.x_s_info["t"] + (x_d_i_info["t"] - self.x_d_0_info["t"])
        else:
            R_new = R_d_i
            delta_new = x_d_i_info["exp"]
            scale_new = self.x_s_info["scale"]
            t_new = x_d_i_info["t"]

        # No tz
        t_new[..., 2].fill_(0)

        # Build new driving kp for decoding
        x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new

        # Optional stitching for stability
        if self.infer_cfg.flag_stitching:
            x_d_i_new = self.wrapper.stitching(self.x_s, x_d_i_new)

        # Decode
        out = self.wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
        rgb = self.wrapper.parse_output(out["out"])[0]  # uint8 RGB, ~256x256
        return rgb
