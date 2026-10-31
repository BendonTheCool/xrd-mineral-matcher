"""Core XRD logic shared by every page.

The preprocessing here must stay identical to the Kaggle notebook (sections 3 and 12),
otherwise uploaded patterns and reference patterns will not be comparable.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

DATA_DIR = Path(os.environ.get("XRD_DATA_DIR", Path(__file__).parent / "data"))

DEFAULT_CONFIG = {
    "wavelength_ref": 1.541838, "tt_min": 5.0, "tt_max": 90.0, "step": 0.05,
    "baseline_window_deg": 2.0, "smooth_sigma_bins": 1.0,
}

WAVELENGTHS = {  # Å
    "Cu Kα (1.5418 Å)": 1.541838,
    "Cu Kα1 (1.5406 Å)": 1.540598,
    "Co Kα (1.7903 Å)": 1.790260,
    "Fe Kα (1.9373 Å)": 1.937355,
    "Cr Kα (2.2910 Å)": 2.291000,
    "Mo Kα (0.7107 Å)": 0.710730,
}

MAX_LINES = 300_000
_NUM_XY = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[,;\s]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


# --------------------------------------------------------------------------- parsing
class PatternError(ValueError):
    """A user-facing problem with an uploaded file."""


def parse_xy_text(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column (2θ, intensity) text file. Header and comment lines are skipped."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    if "\x00" in text:
        raise PatternError("This looks like a binary file. Export the pattern as text with two columns: 2θ and intensity.")
    xs, ys = [], []
    for i, line in enumerate(text.splitlines()):
        if i >= MAX_LINES:
            raise PatternError(f"The file has more than {MAX_LINES:,} lines. Reduce the step size or crop the range.")
        m = _NUM_XY.match(line)
        if m:
            xs.append(float(m.group(1)))
            ys.append(float(m.group(2)))
    x, y = np.asarray(xs), np.asarray(ys)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 50:
        raise PatternError("Found fewer than 50 data points. The file needs two columns of numbers: 2θ in degrees and intensity.")
    if x.min() < 0 or x.max() > 180:
        raise PatternError("The first column must be 2θ in degrees (between 0 and 180).")
    return x, y


# --------------------------------------------------------------------------- preprocessing
@dataclass
class Steps:
    raw_x: np.ndarray
    raw_y: np.ndarray
    cu_x: np.ndarray
    on_grid: np.ndarray       # interpolated, NaN outside measured range
    background: np.ndarray    # NaN outside measured range
    final: np.ndarray         # unit vector on the grid


class Preprocessor:
    def __init__(self, grid: np.ndarray, cfg: dict):
        self.grid = grid
        self.cfg = {**DEFAULT_CONFIG, **cfg}
        self.step = float(self.cfg["step"])
        self.wl_ref = float(self.cfg["wavelength_ref"])

    def to_ref_2theta(self, tt, wl):
        """Convert 2θ measured at wavelength wl to the reference wavelength (same d-spacing, Bragg's law)."""
        tt = np.asarray(tt, float)
        if wl is None or abs(wl - self.wl_ref) < 1e-3:
            return tt
        d = wl / (2 * np.sin(np.radians(tt / 2)))
        s = self.wl_ref / (2 * d)
        out = np.full_like(tt, np.nan)
        ok = (s > 0) & (s < 1)
        out[ok] = 2 * np.degrees(np.arcsin(s[ok]))
        return out

    def finish(self, v, smooth_bins=None):
        smooth_bins = self.cfg["smooth_sigma_bins"] if smooth_bins is None else smooth_bins
        v = np.clip(v, 0, None)
        if smooth_bins:
            v = gaussian_filter1d(v, smooth_bins)
        v = np.sqrt(v)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def run(self, x, y, wl=None, remove_background=True) -> Steps:
        raw_x, raw_y = np.asarray(x, float), np.asarray(y, float)
        cx = self.to_ref_2theta(raw_x, wl)
        ok = np.isfinite(cx) & np.isfinite(raw_y)
        cx2, y2 = cx[ok], raw_y[ok]
        cx2, idx = np.unique(cx2, return_index=True)
        y2 = y2[idx]
        if len(cx2) < 20:
            raise PatternError("After wavelength conversion almost no points are left. Check the wavelength setting.")
        v = np.interp(self.grid, cx2, y2, left=np.nan, right=np.nan)
        inside = np.isfinite(v)
        if inside.sum() < 50:
            raise PatternError(
                f"The pattern barely overlaps the reference range "
                f"({self.grid[0]:.0f}–{self.grid[-1]:.0f}° 2θ for Cu Kα). Check the wavelength and the 2θ column.")
        vi = v[inside]
        win = max(3, int(round(self.cfg["baseline_window_deg"] / self.step)))
        base = uniform_filter1d(minimum_filter1d(vi, win), win) if remove_background else np.zeros_like(vi)
        out = np.zeros_like(self.grid)
        out[inside] = vi - base
        bg = np.full_like(self.grid, np.nan)
        bg[inside] = base
        return Steps(raw_x, raw_y, cx, v, bg, self.finish(out))


# --------------------------------------------------------------------------- search
def top_matches(X: np.ndarray, names: np.ndarray, q: np.ndarray, k: int = 5, exclude: np.ndarray | None = None):
    """Brute-force cosine k-NN (rows of X are unit vectors). Returns [(row, similarity)] for the k best minerals."""
    sims = X @ q.astype(np.float32)
    if exclude is not None:
        sims[exclude] = -np.inf
    order = np.argpartition(-sims, min(500, len(sims) - 1))[:500]
    order = order[np.argsort(-sims[order])]
    seen, out = set(), []
    for i in order:
        if not np.isfinite(sims[i]):
            break
        if names[i] not in seen:
            seen.add(names[i])
            out.append((int(i), float(sims[i])))
            if len(out) == k:
                break
    return out


def shared_peaks(grid, a, b, tol_deg=0.15, n=6):
    """Peaks present in both patterns (within tol). Used to explain *why* a match was suggested."""
    def peaks(v):
        p, props = find_peaks(v, prominence=0.15 * v.max() if v.max() > 0 else 1)
        return p[np.argsort(-props["prominences"])][:25]
    pa, pb = peaks(a), peaks(b)
    if len(pa) == 0 or len(pb) == 0:
        return []
    out = []
    for i in pa:
        j = pb[np.argmin(np.abs(grid[pb] - grid[i]))]
        if abs(grid[j] - grid[i]) <= tol_deg:
            out.append(round(float(grid[i]), 2))
    return sorted(out[:n])


def confidence_label(sim: float, cal: dict) -> tuple[str, str]:
    """Map a similarity to High / Medium / Low using thresholds calibrated on the evaluation results."""
    if sim >= cal["high"]:
        return "High", "ok"
    if sim >= cal["medium"]:
        return "Medium", "warn"
    return "Low", "bad"


def calibrate(per_query) -> dict:
    """Lowest similarity at which accepted answers are ≥90 % (High) / ≥70 % (Medium) correct."""
    fallback = {"high": 0.90, "medium": 0.80, "source": "default thresholds (no evaluation file found)"}
    if per_query is None or len(per_query) < 50:
        return fallback
    s = per_query["sim1"].to_numpy(float)
    c = per_query["correct"].astype(bool).to_numpy()
    order = np.argsort(-s)
    s, c = s[order], c[order]
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)       # accuracy of answers with sim ≥ s[i]

    def thr(target):
        ok = np.where(acc >= target)[0]
        ok = ok[ok >= 20]                                 # need some support
        return float(s[ok.max()]) if len(ok) else None

    hi, med = thr(0.90), thr(0.70)
    if hi is None or med is None:
        return fallback
    return {"high": hi, "medium": min(med, hi), "source": f"calibrated on {len(s):,} held-out test patterns"}


# --------------------------------------------------------------------------- data loading
def load_library(data_dir: Path = DATA_DIR):
    """Load the notebook's exported reference library. Returns None if it is missing."""
    import pandas as pd

    npz, csv = data_dir / "xrd_reference.npz", data_dir / "xrd_reference_meta.csv"
    if not (npz.exists() and csv.exists()):
        return None
    with np.load(npz, allow_pickle=False) as z:
        X, grid = z["X"].astype(np.float32), z["grid"].astype(float)
    meta = pd.read_csv(csv, keep_default_na=False, na_values=[""])
    if len(meta) != len(X):
        raise RuntimeError(f"xrd_reference_meta.csv has {len(meta)} rows but xrd_reference.npz has {len(X)} patterns.")
    cfg_path = data_dir / "xrd_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    for col in ("rruff_id", "chemistry", "crystal_system", "kind", "sample"):
        if col not in meta:
            meta[col] = ""
        meta[col] = meta[col].fillna("").astype(str)
    return {"X": X, "grid": grid, "meta": meta, "cfg": cfg, "names": meta["name"].astype(str).to_numpy()}


def load_metrics(data_dir: Path = DATA_DIR):
    import pandas as pd

    m = json.loads((data_dir / "metrics.json").read_text()) if (data_dir / "metrics.json").exists() else None
    pq = pd.read_csv(data_dir / "evaluation_per_query.csv") if (data_dir / "evaluation_per_query.csv").exists() else None
    return m, pq


def tidy_formula(s: str) -> str:
    """RRUFF writes subscripts as _2_ ; turn them into Unicode subscripts."""
    sub = str.maketrans("0123456789.", "₀₁₂₃₄₅₆₇₈₉.")
    return re.sub(r"_([\d.]+)_", lambda m: m.group(1).translate(sub), s or "")
