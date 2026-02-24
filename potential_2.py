# potential_2.py
# 2025/9/18
# 2025/11/10 revised
# 2026/2/24 patched
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
from pathlib import Path
from typing import Dict, Literal, Optional

import math
import sys
import types

import joblib
import numpy as np
from scipy.interpolate import RegularGridInterpolator


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
    shape: SpeciesShape = "spherical"


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
            if theta_points is None:
                raise ValueError(f"{e1}-{e2}: theta_points is required for dim=2")
            self.axes = (np.asarray(r_points), np.asarray(theta_points))
        else:
            self.dim = 3
            if theta_points is None or phi_points is None:
                raise ValueError(f"{e1}-{e2}: theta_points and phi_points are required for dim=3")
            self.axes = (np.asarray(r_points), np.asarray(theta_points), np.asarray(phi_points))

        self.energies = np.asarray(energies)

        # 形状整合性チェック
        expected = tuple(len(ax) for ax in self.axes)
        if self.energies.shape != expected:
            raise ValueError(
                f"{e1}-{e2}: energies shape mismatch. "
                f"expected {expected}, got {self.energies.shape}"
            )

        self.interp = RegularGridInterpolator(
            self.axes,
            self.energies,
            bounds_error=False,
            fill_value=None if extrapolation else np.nan,
        )

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
    
        # scipy の挙動差で val がスカラーになる場合があるため両対応
        if np.ndim(val) == 0:
            return float(val)
    
        v0 = val[0]
    
        # v0 がスカラーならOK
        if np.ndim(v0) == 0:
            return float(v0)
    
        # ここに来るのは energies 次元と dim 判定がズレているとき
        raise TypeError(
            f"{self.e1}-{self.e2}: interpolator returned non-scalar. "
            f"type={type(v0)}, shape={np.shape(v0)}"
        )

    def energy(self, r: float, theta: Optional[float] = None, phi: Optional[float] = None) -> float:
        if self.dim == 1:
            return self._eval_rgi(float(r))
        if self.dim == 2:
            t, _ = self._normalize_angles(theta, None)
            return self._eval_rgi(float(r), t)
        t, p = self._normalize_angles(theta, phi)
        return self._eval_rgi(float(r), t, p)

    def __repr__(self) -> str:
        return f"PairPotential({self.e1}-{self.e2}, dim={self.dim})"


# =============================
# 3. レジストリ
# =============================
class PotentialRegistry:
    def __init__(self, species_info: Dict[str, SpeciesInfo]):
        self.species_info = species_info
        self._pairs: Dict[tuple[str, str], PairPotential] = {}

    @staticmethod
    def _pair_key(e1: str, e2: str):
        return (e1, e2) if e1 <= e2 else (e2, e1)

    def energy(self, e1: str, e2: str, *args) -> float:
        key = self._pair_key(e1, e2)
        pot = self._pairs[key]

        r = args[0] if len(args) > 0 else None
        th = args[1] if len(args) > 1 else None
        ph = args[2] if len(args) > 2 else None
        if r is None:
            raise ValueError("At least r must be provided.")

        rr = float(r)

        # 距離クランプ
        rmin = None
        rmax = None

        for amin, amax in [("rmin", "rmax"), ("r_min", "r_max"), ("min_r", "max_r")]:
            if hasattr(pot, amin) and hasattr(pot, amax):
                rmin = float(getattr(pot, amin))
                rmax = float(getattr(pot, amax))
                break

        if (rmin is None or rmax is None) and hasattr(pot, "r_grid"):
            try:
                rg = list(getattr(pot, "r_grid"))
                rmin = float(min(rg))
                rmax = float(max(rg))
            except Exception:
                pass

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

        if val is None or (not math.isfinite(float(val))):
            return 1.0e6

        return float(val)

    def get_energy(self, e1: str, e2: str, *args) -> float:
        return self.energy(e1, e2, *args)

    def register_npz(self, e1: str, e2: str, npz_path: str | Path):
        npz_path = Path(npz_path)
        data = np.load(npz_path)

        # あなたのnpz形式に合わせる
        r_points = data["r"]
        energies = data["energies"]
        theta_points = data["theta"] if "theta" in data.files else None
        phi_points = data["phi"] if "phi" in data.files else None

        # 今回の探索では axial は MA と FA のみ
        axial = {"MA", "FA"}

        # species_info が空なら最小生成
        if not self.species_info:
            def shape(x): return "axial" if x in axial else "spherical"
            self.species_info = {e1: SpeciesInfo(shape(e1)), e2: SpeciesInfo(shape(e2))}

        # 未登録種があれば追加
        if e1 not in self.species_info:
            self.species_info[e1] = SpeciesInfo("spherical")
        if e2 not in self.species_info:
            self.species_info[e2] = SpeciesInfo("spherical")

        # ここが重要: 既に入っていても強制上書き
        if e1 in axial:
            self.species_info[e1].shape = "axial"
        if e2 in axial:
            self.species_info[e2].shape = "axial"

        pot = PairPotential(
            e1=e1,
            e2=e2,
            species_info=self.species_info,
            r_points=r_points,
            energies=energies,
            theta_points=theta_points,
            phi_points=phi_points,
        )
        self._pairs[self._pair_key(e1, e2)] = pot


# =============================
# 旧pickle互換シム
# =============================
def _install_legacy_main_shim(module_name="__main__"):
    mod = sys.modules.get(module_name)
    if mod is None or not hasattr(mod, "PairPotential"):
        shim = types.ModuleType(module_name)
        shim.PairPotential = PairPotential
        shim.PotentialRegistry = PotentialRegistry
        shim.SpeciesInfo = SpeciesInfo
        sys.modules[module_name] = shim


# =============================
# 4. 保存・ロードユーティリティ
# =============================
def save_registry(registry: PotentialRegistry, filepath: str | Path = "potentials.pkl") -> None:
    joblib.dump(registry, str(filepath))
    print(f"Registry saved to {filepath}")


def load_registry(filepath: str | Path = "potentials.pkl", npz_dir: Optional[str] = None) -> PotentialRegistry:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"{filepath} not found. cwd={Path.cwd()}")

    _install_legacy_main_shim("__main__")

    try:
        registry = joblib.load(str(path))
        print(f"Registry loaded from {filepath}")

        # pkl由来のspecies_infoが古い可能性があるので強制補正
        axial = {"MA", "FA"}
        if hasattr(registry, "species_info") and isinstance(registry.species_info, dict):
            for sp in axial:
                if sp in registry.species_info:
                    registry.species_info[sp].shape = "axial"

        # energies が無い旧形式なら interp.values から復元を試す
        for pot in registry._pairs.values():
            if not hasattr(pot, "energies"):
                if hasattr(pot, "interp") and hasattr(pot.interp, "values"):
                    pot.energies = np.asarray(pot.interp.values)
                else:
                    raise RuntimeError(
                        "This pickle was created with an old PairPotential "
                        "that does not store energies and cannot be upgraded."
                    )

            if not hasattr(pot, "axes"):
                raise RuntimeError("Old pickle is missing axes, cannot rebuild interpolator.")

            pot.interp = RegularGridInterpolator(
                pot.axes,
                pot.energies,
                bounds_error=False,
                fill_value=None,
            )

        print("[patch] Rebuilt interpolators after pkl load")
        return registry

    except Exception as e:
        msg = f"[WARN] Failed to load usable pkl ({e})."
        if npz_dir is None:
            raise RuntimeError(
                msg + " The pickle exists but is incompatible. "
                      "Either re-save using classes from potential_2.py, "
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
