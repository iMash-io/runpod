# app/wav2lip_node.py
import sys
import time
import queue
import threading
from typing import Optional, List

import cv2
import numpy as np
import torch
import torchaudio
import librosa

class Wav2LipNode:
    """
    Streaming audio->lip node for a *single portrait image*.
    - Feed PCM16 audio (any sample rate); it's resampled to 16kHz internally
    - Pull frames() to get RGB frames (~256x256 or upscaled)
    - Target ~25 FPS; latency ~240-320 ms depending on chunking
    """

    def __init__(self, portrait_bgr: np.ndarray, out_size: int = 512, device: str = "cuda",
                 checkpoint_path: str = "/app/wav2lip/checkpoints/wav2lip_gan.pth",
                 fps: int = 25):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.out_size = out_size
        self.target_fps = fps

        # --- Import Wav2Lip modules
        root = "/app/wav2lip"
        if root not in sys.path:
            sys.path.append(root)
        from inference import load_model
        from models import Wav2Lip

        # --- Prepare model
        self.model: torch.nn.Module = load_model(checkpoint_path, cpu=(self.device == "cpu"))
        self.model = self.model.to(self.device).eval()

        # --- Prepare image (face crop -> 96x96 mouth region is ideal; we keep it simple: center crop)
        self.src_rgb_256 = self._prepare_src_256(portrait_bgr)

        # --- Audio buffers
        self.lock = threading.Lock()
        self.audio_buf = np.zeros((0,), dtype=np.int16)
        self.frames_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)

        # --- Worker thread: consume audio, produce frames
        self.worker = threading.Thread(target=self._run_loop, daemon=True)
        self.worker.start()

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

    def _prepare_src_256(self, bgr: np.ndarray) -> np.ndarray:
        sq = self._center_square_resize_bgr(bgr, 256)
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        return rgb

    def push_pcm16(self, pcm: np.ndarray, sample_rate: int):
        """Append PCM16 mono or stereo audio. Resample to 16k inside the worker."""
        if pcm.ndim == 2:  # stereo -> mono
            pcm = pcm.mean(axis=1).astype(np.int16)
        with self.lock:
            self.audio_buf = np.concatenate([self.audio_buf, pcm.astype(np.int16)])

    def pop_frame_nowait(self) -> Optional[np.ndarray]:
        try:
            return self.frames_q.get_nowait()
        except queue.Empty:
            return None

    # ---------- Core loop ----------
    def _run_loop(self):
        """
        Pull ~320ms (5120 samples at 16k) windows, generate ~8 frames at 25 FPS,
        push frames into queue. Uses simple overlap for continuity.
        """
        target_sr = 16000
        hop_s = 1.0 / self.target_fps
        win_s = 0.32   # seconds per inference window
        hop_samples = int(hop_s * target_sr)       # ~640 samples @16k
        win_samples = int(win_s * target_sr)       # ~5120 samples @16k
        last_residual = np.zeros(0, dtype=np.float32)

        # Precompute face batch (static image repeated as needed by model)
        face_batch = torch.from_numpy(self.src_rgb_256 / 255.).permute(2, 0, 1).unsqueeze(0)  # 1,3,256,256
        face_batch = face_batch.to(self.device, dtype=torch.float32)

        while True:
            # 1) collect enough audio
            with self.lock:
                if len(self.audio_buf) < hop_samples:
                    time.sleep(0.005)
                    continue
                buf = self.audio_buf.copy()
                self.audio_buf = np.zeros((0,), dtype=np.int16)

            # 2) resample to 16k and accumulate
            wav = torch.from_numpy(buf.astype(np.float32) / 32768.0)
            if wav.ndim == 0:
                wav = wav.unsqueeze(0)
            wav16 = torchaudio.functional.resample(wav, orig_freq=target_sr if False else 24000, new_freq=target_sr) \
                if False else wav  # set False->True if you know incoming rate != 16k
            # Fallback: assume unknown sample rate & resample with librosa (slower, safer)
            wav16 = librosa.resample(wav16.numpy(), orig_sr=target_sr, target_sr=target_sr) if not isinstance(wav16, torch.Tensor) else wav16.numpy()

            # 3) concatenate with residual
            audio_f = np.concatenate([last_residual, wav16.astype(np.float32)])
            # 4) if enough for at least one hop, produce frames
            ptr = 0
            while ptr + win_samples <= len(audio_f):
                chunk = audio_f[ptr:ptr+win_samples]  # 5120 samples @16k
                ptr += hop_samples * 8  # ~8 frames per window

                # Build mel for this chunk (Wav2Lip expects 80-d mel)
                mel = librosa.feature.melspectrogram(
                    y=chunk, sr=target_sr, n_fft=1024, hop_length=160, win_length=400, n_mels=80, fmin=0, fmax=8000)
                mel = np.log(np.maximum(1e-5, mel))
                mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)  # 1,1,80,T

                # Model expects a face batch and mel; it will output a sequence of frames
                with torch.no_grad():
                    # output: [B,T,3,96,96] or similar, depending on repo version
                    pred = self.model(face_batch, mel)
                # Convert to HWC uint8 RGB frames (resize to out_size)
                frames = pred.squeeze(0).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()  # T,96,96,3
                for fr in frames:
                    fr_u8 = (fr * 255.0).astype(np.uint8)
                    fr_u8 = cv2.resize(fr_u8, (self.out_size, self.out_size), interpolation=cv2.INTER_CUBIC)
                    try:
                        self.frames_q.put_nowait(fr_u8)
                    except queue.Full:
                        pass

            # leftover
            last_residual = audio_f[ptr:]
