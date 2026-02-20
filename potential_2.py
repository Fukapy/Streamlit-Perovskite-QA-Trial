# potentials_2.py
# 2025/9/18
# 2025/11/10 revised
# Ryo Fukasawa
# -*- coding: utf-8 -*-
"""
Pairwise potential registry with fast interpolation/extrapolation.

- Save a fitted registry as a single .pkl (joblib) and reuse it without .npz files
- Order-free species pair: registry.energy("I","MA", 3.5, 37.0, 180) works
- Positional args only: (r, theta, phi)
- Unused angles can be None/NaN/any number (ignored as needed)
- Raises a clear error if an unregistered pair/species is requested

Requirements:
    pip install numpy scipy joblib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import joblib

# potential.py の先頭付近に追加
import sys, types

def _install_legacy_main_shim(module_name="__main__"):
    """
    旧pickle（__main__.PairPotential など）を読めるように、
    現在のモジュール内クラスを __main__ 配下にも見せるシムを挿入。
    """
    mod = sys.modules.get(module_name)
    if mod is None or not hasattr(mod, "PairPotential"):
        shim = types.ModuleType(module_name)
        # この potential.py 内のクラスを参照させる
        shim.PairPotential = PairPotential
        shim.PotentialRegistry = PotentialRegistry
        shim.SpeciesInfo = SpeciesInfo
        sys.modules[module_name] = shim




__all__ = [
    "SpeciesInfo",
    "PairPotential",
    "PotentialRegistry",
    "save_registry",
    "load_registry",
    "register_npz_from_dir",
]

# =============================
# 1. Species 情報
# =============================
SpeciesShape = Literal["spherical", "axial"]

@dataclass
class SpeciesInfo:
    """Species metadata (minimal)."""
    shape: SpeciesShape = "spherical"  # "spherical" or "axial"

# =============================
# 2. ペアポテンシャル
# =============================
class PairPotential:
    def __init__(
        self,
        e1: str,
        e2: str,
        species_info: Dict[str, SpeciesInfo],
        r_points: np.ndarray,
        energies: np.ndarray,
        theta_points: Optional[np.ndarray] = None,
        phi_points: Optional[np.ndarray] = None,
        assume_theta_mirror: bool = True,
        extrapolation: bool = True,
    ):
        self.e1, self.e2 = e1, e2
        self.species_info = species_info
        self.shape1 = species_info[e1].shape
        self.shape2 = species_info[e2].shape
        self.assume_theta_mirror = assume_theta_mirror

        if self.shape1 == "spherical" and self.shape2 == "spherical":
            self.dim = 1
            self.axes = (np.asarray(r_points),)
        elif (self.shape1 == "axial") ^ (self.shape2 == "axial"):
            self.dim = 2
            self.axes = (np.asarray(r_points), np.asarray(theta_points))
        else:
            self.dim = 3
            self.axes = (np.asarray(r_points), np.asarray(theta_points), np.asarray(phi_points))

        # ★ energies を明示的に保存
        self.energies = np.asarray(energies)

        self.interp = RegularGridInterpolator(
            self.axes,
            self.energies,
            bounds_error=False,
            fill_value=None if extrapolation else np.nan,
        )

    # --- 角度処理 ---
    @staticmethod
    def _wrap_angle(theta: float, period: float) -> float:
        t = theta % period
        return 0.0 if np.isclose(t, period) else t

    @staticmethod
    def _mirror_180(theta: float) -> float:
        t = theta % 360.0
        return 360.0 - t if t > 180.0 else t

    def _normalize_angles(self, theta: Optional[float], phi: Optional[float]):
        t, p = theta, phi
        if self.dim >= 2 and t is not None:
            t = self._mirror_180(float(t)) if self.assume_theta_mirror else self._wrap_angle(float(t), 180.0)
        if self.dim == 3 and p is not None:
            p = self._wrap_angle(float(p), 360.0)
        return t, p

    def _eval_rgi(self, *coords: float) -> float:
        pt = np.array(coords, dtype=float, ndmin=1).reshape(1, -1)
        val = self.interp(pt)
        return float(val[0])

    def energy(self, r: float, theta: Optional[float] = None, phi: Optional[float] = None) -> float:
        if self.dim == 1:
            return self._eval_rgi(float(r))
        elif self.dim == 2:
            t, _ = self._normalize_angles(theta, None)
            return self._eval_rgi(float(r), t)
        else:
            t, p = self._normalize_angles(theta, phi)
            return self._eval_rgi(float(r), t, p)

    def __repr__(self) -> str:
        return f"PairPotential({self.e1}-{self.e2}, dim={self.dim})"

# =============================
# 3. レジストリ　2026/2/2
# =============================
import math

class PotentialRegistry:
    def __init__(self, species_info):
        self.species_info = species_info
        self._pairs = {}

    @staticmethod
    def _pair_key(e1, e2):
        return (e1, e2) if e1 <= e2 else (e2, e1)

    def energy(self, e1, e2, *args) -> float:
        key = self._pair_key(e1, e2)
        pot = self._pairs[key]

        r = args[0] if len(args) > 0 else None
        th = args[1] if len(args) > 1 else None
        ph = args[2] if len(args) > 2 else None
        if r is None:
            raise ValueError("At least r must be provided.")

        rr = float(r)

        # pot から距離範囲を推定してクランプ
        rmin = None
        rmax = None

        # よくある属性名
        for amin, amax in [("rmin", "rmax"), ("r_min", "r_max"), ("min_r", "max_r")]:
            if hasattr(pot, amin) and hasattr(pot, amax):
                rmin = float(getattr(pot, amin))
                rmax = float(getattr(pot, amax))
                break

        # 距離グリッドを持つ場合
        if (rmin is None or rmax is None) and hasattr(pot, "r_grid"):
            try:
                rg = list(getattr(pot, "r_grid"))
                rmin = float(min(rg))
                rmax = float(max(rg))
            except Exception:
                pass

        # 補間器が距離グリッドを持つ場合
        if (rmin is None or rmax is None) and hasattr(pot, "r_values"):
            try:
                rg = list(getattr(pot, "r_values"))
                rmin = float(min(rg))
                rmax = float(max(rg))
            except Exception:
                pass

        if rmin is not None and rmax is not None:
            eps = 1e-9
            if rr < rmin:
                rr = rmin + eps
            elif rr > rmax:
                rr = rmax - eps

        val = pot.energy(rr, th, ph)

        # nan や inf を絶対に返さない
        if val is None or (not math.isfinite(float(val))):
            # QUBO を壊さないための罰則値
            # lam_onehot や lam_comp より十分大きい値にしておく
            return 1.0e6

        return float(val)

    def get_energy(self, e1: str, e2: str, *args) -> float:
        return self.energy(e1, e2, *args)
        

# =============================
# 4. 保存・ロードユーティリティ
# =============================
def save_registry(registry: PotentialRegistry, filepath: str | Path = "potentials.pkl") -> None:
    joblib.dump(registry, str(filepath))
    print(f"Registry saved to {filepath}")

# 既存の load_registry を以下で置き換え
def load_registry(filepath: str | Path = "potentials.pkl", npz_dir: Optional[str] = None) -> PotentialRegistry:
    """
    まず .pkl をロード。__main__ で保存された古いpickleにも対応する。
    読み込み後は RegularGridInterpolator を再構築して互換性を確保。
    .pkl が壊れている/互換不可なら npz_dir から再生成（指定時）。
    """
    path = Path(filepath)

    # 1) まず存在確認（相対パスの取り違いを早期検知）
    if not path.exists():
        # ユーザに分かりやすい絶対パスも出す
        raise FileNotFoundError(f"{filepath} not found. cwd={Path.cwd()}")

    # 2) 旧pickle互換のためのシムを挿入（__main__.PairPotential 等の解決）
    _install_legacy_main_shim("__main__")

    # 3) joblib.load を試す
    try:
        registry = joblib.load(str(path))
        print(f"Registry loaded from {filepath}")

        # interp を再構築（scipy バージョン差吸収）
        for pot in registry._pairs.values():
            axes = pot.axes
            # scipy<=1.10系と>=1.11系の両対応
            values = getattr(pot.interp, "values", None)
            if values is None:
                values = getattr(pot.interp, "_values", None)
            if values is None:
                # 万一取り出せなければ、PairPotential が持つ元配列にフォールバック
                # （保存時に参照を残していない場合は例外にする）
                raise RuntimeError("Failed to extract interpolator values for rebuild.")
            pot.interp = RegularGridInterpolator(
                axes,
                values,
                bounds_error=False,
                fill_value=None
            )
        print("[patch] Rebuilt interpolators after pkl load")
        return registry

    except Exception as e:
        msg = f"[WARN] Failed to load usable pkl ({e})."
        if npz_dir is None:
            # 元の実装だと FileNotFoundError を投げ直してしまい誤解を招くので、
            # ここでは元例外を添えて明確に案内する
            raise RuntimeError(
                msg + " The pickle exists but is incompatible. "
                      "Either re-save using classes from potential.py, "
                      "or specify npz_dir to rebuild."
            ) from e

        print(msg + " Rebuilding from npz_dir...")
        registry = PotentialRegistry(species_info={})
        register_npz_from_dir(registry, npz_dir)
        save_registry(registry, filepath)
        return registry

# =============================
# 5. ディレクトリ一括登録
# =============================
def register_npz_from_dir(registry: PotentialRegistry, directory: str | Path) -> None:
    directory = Path(directory)
    files = sorted(directory.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {directory}")
        return
    for filepath in files:
        name = filepath.stem
        try:
            e1, e2 = name.split("-")
        except ValueError:
            print(f"Skip {filepath} (bad filename)")
            continue
        registry.register_npz(e1, e2, filepath)
        print(f"Registered {e1}-{e2} from {filepath}")
