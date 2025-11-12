# ===== step_logger.py =====
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import json
import numpy as np
import re

try:
    from PIL import Image
except ImportError:
    Image = None  # If you don't want frames, you can skip saving images.

import torch
import torch.nn.functional as F


# Fixed 3-bit action space -> 8 actions
# Bit order: [Right, Jump, Left] to match your index_to_multibinary()
# Adjust names or bit order here if your mapping differs.
ACTION_BITS = ["Left", "Jump", "Right"]


def multibinary_to_label(bits: np.ndarray) -> str:
    """Turn [r, j, l] like [1,1,0] into 'Right+Jump' (or 'Idle' if all zeros)."""
    on = [name for name, b in zip(ACTION_BITS, bits) if int(b) == 1]
    return "Idle" if not on else "+".join(on)


def all_action_labels() -> List[str]:
    labels = []
    for idx in range(8):
        b = np.array([(idx >> 2) & 1, (idx >> 1) & 1, idx & 1])  # [r, j, l]
        labels.append(multibinary_to_label(b))
    return labels


ALL_LABELS = all_action_labels()  # index-aligned list of 8 labels


class EpisodeLogger:
    """
    Collects per-frame logs and writes a JSON shaped like:

    {
      "run": "<run_id>",
      "epsilon": <float>,
      "policy": "q_softmax",
      "frames": [
        {
          "step": 0,
          "x": 113,
          "y": 456,
          "frame_path": "captures\\<run>\\<prefix>_frame_000000.png",
          "probs": { "<label>": <float>, ... },
          "actions": "Right+Jump"
        },
        ...
      ]
    }

    Overwrite control:
      - unique=True  -> use a fresh folder if one exists: captures/<run_id>_001, _002, ...
      - resume=True  -> append frames in the same folder, continuing the index after the
                        highest existing frame number for this prefix.
    """

    def __init__(
        self,
        run_id: str,
        epsilon: float,
        capture_dir: str = "captures",
        policy: str = "q_softmax",
        output_path: Optional[str] = None,
        image_format: str = "png",
        frame_prefix: Optional[str] = None,
        unique: bool = False,
        resume: bool = False,
    ):
        self.run_id = run_id
        self.epsilon = float(epsilon)
        self.policy = policy
        self.frames: List[Dict] = []

        # Where to store images:
        self.capture_root = Path(capture_dir)
        self.capture_dir = self.capture_root / run_id

        # Ensure per-run uniqueness if requested
        if unique:
            base = self.capture_dir
            i = 1
            while self.capture_dir.exists():
                self.capture_dir = Path(f"{base}_{i:03d}")
                i += 1

        self.capture_dir.mkdir(parents=True, exist_ok=True)

        self.image_format = image_format.lower()
        self.frame_prefix = frame_prefix or run_id  # also used in filenames

        # JSON output path (defaults to inside the run folder)
        if output_path is None:
            self.output_path = self.capture_dir / f"{self.run_id}.json"
        else:
            self.output_path = Path(output_path)

        if Image is None:
            print("[EpisodeLogger] Pillow not installed. Frame PNGs will be skipped.")

        # When resuming, compute the starting offset to avoid overwriting
        self._step_offset = 0
        if resume:
            self._step_offset = self._find_max_existing_index() + 1

    # ---------- internals ----------

    def _find_max_existing_index(self) -> int:
        """Scan capture_dir and find the largest frame index for this prefix."""
        # Matches <prefix>_frame_000123.png
        pat = re.compile(
            re.escape(f"{self.frame_prefix}_frame_") + r"(\d{6})\." + re.escape(self.image_format) + r"$"
        )
        max_idx = -1
        for p in self.capture_dir.glob(f"{self.frame_prefix}_frame_*.{self.image_format}"):
            m = pat.match(p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx

    def _save_frame_image(self, step: int, frame: Optional[np.ndarray]) -> str:
        """
        Save frame to captures\<run_id>\<prefix>_frame_XXXXXX.png (Windows backslashes).
        If frame is None or Pillow missing, still return the intended path string.
        """
        idx = step + self._step_offset  # prevents overwriting when resuming
        fname = f"{self.frame_prefix}_frame_{idx:06d}.{self.image_format}"
        fpath = self.capture_dir / fname

        if frame is not None and Image is not None:
            # Expect HxWxC uint8 RGB. Convert if needed.
            arr = frame if isinstance(frame, np.ndarray) else np.array(frame)

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            if arr.ndim == 2:
                # grayscale -> RGB
                arr = np.stack([arr, arr, arr], axis=-1)

            Image.fromarray(arr).save(fpath)

        # Return Windows-style path (for your schema)
        return str(fpath).replace("/", "\\")

    @torch.no_grad()
    def compute_probs_from_q(
        self,
        q_values: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Given 8 Q-values, return softmax probs over the 8 combined-action labels.
        q_values: 1D tensor of shape [8]
        """
        if q_values.ndim != 1 or q_values.numel() != 8:
            raise ValueError("q_values must be a 1D tensor with 8 elements.")

        logits = q_values / float(max(temperature, 1e-8))
        probs = F.softmax(logits, dim=0).cpu().numpy().astype(float)
        return {label: float(p) for label, p in zip(ALL_LABELS, probs)}

    # ---------- public API ----------

    def log_step(
        self,
        step: int,
        action_index: int,
        q_values: Optional[torch.Tensor] = None,
        frame: Optional[np.ndarray] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ):
        """
        - action_index: 0..7
        - q_values: torch tensor [8] from your network (before softmax)
        - frame: RGB array for saving. If None, image saving is skipped.
        - x/y: optional coordinates (falls back to None if not provided)
        """
        if not (0 <= action_index < 8):
            raise ValueError("action_index must be in [0,7]")

        # Save image (or synthesize a path even if None)
        frame_path = self._save_frame_image(step, frame)

        # Convert to label for this combined action
        action_label = ALL_LABELS[action_index]

        # Probs dict if q_values provided
        probs_dict = None
        if q_values is not None:
            probs_dict = self.compute_probs_from_q(q_values)

        record = {
            "step": int(step),
            "x": int(x) if x is not None else None,
            "y": int(y) if y is not None else None,
            "frame_path": frame_path,
            "probs": probs_dict,
            "actions": action_label,
        }
        self.frames.append(record)

    def dump(self) -> Dict:
        return {
            "run": self.run_id,
            "epsilon": self.epsilon,
            "policy": self.policy,
            "frames": self.frames,
        }

    def save(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        payload = self.dump()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=ensure_ascii, indent=indent)
        return str(self.output_path)