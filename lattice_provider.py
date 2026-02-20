# lattice_provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class LatticeResult:
    mode: str  # "fixed" | "lookup" | "ml" | "fallback"
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    note: str = ""


def _normalize(fracs: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in fracs.values())
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in fracs.items() if float(v) > 0}


def lattice_fixed(a_default: float = 6.30) -> LatticeResult:
    return LatticeResult(
        mode="fixed",
        a=a_default, b=a_default, c=a_default,
        alpha=90.0, beta=90.0, gamma=90.0,
        note="fixed cubic cell"
    )


def lattice_lookup_table(
    A_fracs: Dict[str, float],
    B_fracs: Dict[str, float],
    X_fracs: Dict[str, float],
    table: Dict[Tuple[str, str, str], float],
    a_default: float = 6.30,
) -> LatticeResult:
    """
    まずは単純に ABX の完全一致だけ見る。
    混晶の場合は、主要成分（最大比率）だけで代表させる。
    """
    A = _normalize(A_fracs)
    B = _normalize(B_fracs)
    X = _normalize(X_fracs)
    if not A or not B or not X:
        return lattice_fixed(a_default=a_default)

    a_main = max(A.items(), key=lambda kv: kv[1])[0]
    b_main = max(B.items(), key=lambda kv: kv[1])[0]
    x_main = max(X.items(), key=lambda kv: kv[1])[0]

    key = (a_main, b_main, x_main)
    if key in table:
        a = float(table[key])
        return LatticeResult(
            mode="lookup",
            a=a, b=a, c=a,
            alpha=90.0, beta=90.0, gamma=90.0,
            note=f"lookup hit {key}"
        )

    return LatticeResult(
        mode="fallback",
        a=a_default, b=a_default, c=a_default,
        alpha=90.0, beta=90.0, gamma=90.0,
        note=f"lookup miss {key}, use default"
    )


def lattice_from_user_or_default(
    user_a: Optional[float],
    A_fracs: Dict[str, float],
    B_fracs: Dict[str, float],
    X_fracs: Dict[str, float],
    a_default: float = 6.30,
    enable_lookup: bool = True,
    lookup_table: Optional[Dict[Tuple[str, str, str], float]] = None,
) -> LatticeResult:
    """
    優先順位
    1) user_a があればそれを採用
    2) enable_lookup かつ table があれば lookup
    3) それ以外は固定値
    """
    if user_a is not None:
        a = float(user_a)
        return LatticeResult(
            mode="user",
            a=a, b=a, c=a,
            alpha=90.0, beta=90.0, gamma=90.0,
            note="user specified"
        )

    if enable_lookup and lookup_table is not None:
        return lattice_lookup_table(A_fracs, B_fracs, X_fracs, lookup_table, a_default=a_default)

    return lattice_fixed(a_default=a_default)

import csv
from pathlib import Path
from typing import List


def _a_pc_from_abc(a: float, b: float, c: float) -> float:
    return (a * b * c) ** (1.0 / 3.0)


def build_lookup_table_from_tilley_csv(csv_path: str) -> dict[tuple[str, str, str], float]:
    """
    Tilley_Perovskite_Crystals.csv から (A,B,X) -> 平均 a_pc を作る。
    CSVに同じ(A,B,X)が複数行ある場合は a_pc の平均。
    """
    p = Path(csv_path)
    if not p.exists():
        return {}

    acc: dict[tuple[str, str, str], List[float]] = {}

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            A = (row.get("A site") or "").strip()
            B = (row.get("B site") or "").strip()
            X = (row.get("X site") or "").strip()
            if not (A and B and X):
                continue

            try:
                a = float((row.get("a") or "").strip())
                b = float((row.get("b") or "").strip())
                c = float((row.get("c") or "").strip())
            except Exception:
                continue

            a_pc = _a_pc_from_abc(a, b, c)
            key = (A, B, X)
            acc.setdefault(key, []).append(a_pc)

    table: dict[tuple[str, str, str], float] = {}
    for k, vals in acc.items():
        table[k] = sum(vals) / len(vals)

    return table


def lattice_predict_simple_from_csv(
    A_fracs: Dict[str, float],
    B_fracs: Dict[str, float],
    X_fracs: Dict[str, float],
    csv_path: str,
    a_default: float = 6.30,
) -> LatticeResult:
    """
    簡易推論: CSV由来ルックアップ。
    混晶は主要成分で代表させる。
    """
    table = build_lookup_table_from_tilley_csv(csv_path)
    if not table:
        return lattice_fixed(a_default=a_default)

    return lattice_lookup_table(
        A_fracs=A_fracs,
        B_fracs=B_fracs,
        X_fracs=X_fracs,
        table=table,
        a_default=a_default,
    )
