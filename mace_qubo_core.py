# 20260203

### 0．分子スケーリング

import os
import numpy as np
from itertools import product
import datetime
from amplify import VariableGenerator, FixstarsClient, solve
from potential_2 import load_registry

# --- Amplify 互換ユーティリティ（API差分吸収） ---
def _gen_array_compat(gen: VariableGenerator, name: str, *shape):
    """
    amplify のバージョン差により gen.array のシグネチャが異なるため吸収する。
    旧: gen.array("qA", n, m)
    新: gen.array(VariableType.Binary, n, m) など
    """
    try:
        return gen.array(name, *shape)
    except Exception:
        try:
            from amplify import VariableType
            return gen.array(VariableType.Binary, *shape)
        except Exception:
            try:
                from amplify import Binary
                return gen.array(Binary, *shape)
            except Exception as e:
                raise e


def _registry_energy_compat(registry, e1: str, e2: str, r: float) -> float:
    """
    potential_2.PotentialRegistry の互換層。
    旧: registry.get_energy(e1, e2, r)
    新: registry.energy(e1, e2, r) など
    """
    if hasattr(registry, "get_energy"):
        return float(registry.get_energy(e1, e2, r))
    if hasattr(registry, "energy"):
        return float(registry.energy(e1, e2, r))
    if hasattr(registry, "__call__"):
        # 念のため: registry(e1, e2, r) の形式を許容
        return float(registry(e1, e2, r))
    raise AttributeError("registry has neither get_energy nor energy")


def _safe_pair_energy(registry, label: str, e1: str, e2: str, r: float, penalty: float = 1.0e6) -> float:
    """
    NaN/Inf を objective に混入させないための安全取得。
    補間範囲外や欠損により NaN が出た場合は、距離をグリッド範囲にクランプして再評価。
    それでもダメなら大きな罰則値を返す。
    """
    e = _registry_energy_compat(registry, e1, e2, r)
    if np.isfinite(e):
        return float(e)

    # クランプして再評価
    try:
        key = registry._pair_key(e1, e2)  # noqa: SLF001
        pot = registry._pairs[key]        # noqa: SLF001
        rmin = float(pot.axes[0][0])
        rmax = float(pot.axes[0][-1])
        r2 = min(max(float(r), rmin), rmax)
        e2v = _registry_energy_compat(registry, e1, e2, r2)
        if np.isfinite(e2v):
            return float(e2v)
    except Exception:
        pass

    # 最終フォールバック
    # print は大量に出るので必要最小限
    # print(f"[warn] Non-finite energy: {label} pair=({e1},{e2}) r={r} -> penalty")
    return float(penalty)

# MA/FA を分子（8原子）として扱うスケーリング
def scale_energy_molecular(e1, e2, val):
    mols = {"MA", "FA"}
    if (e1 in mols) and (e2 in mols):
        return val / 64.0   # 分子–分子
    elif (e1 in mols) ^ (e2 in mols):
        return val / 8.0    # 分子–単原子
    else:
        return val          # 単原子–単原子


### 1. Glazer tiltの定義 (例: a^0a^0c^- などに相当)

def rotation_matrix_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[ 1., 0., 0.],
                     [ 0., c,-s],
                     [ 0., s, c]])

def rotation_matrix_y(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[ c, 0., s],
                     [ 0., 1., 0.],
                     [-s, 0., c]])

def rotation_matrix_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[ c,-s, 0.],
                     [ s, c, 0.],
                     [ 0., 0., 1.]])

def apply_glazer_tilt(vec, angles_deg_xyz, signs_xyz):
    """
    angles_deg_xyz = (θx, θy, θz) in degrees
    signs_xyz = (sx, sy, sz)  e.g. (0, 0, -1) means c^- tilt only
    """
    thx, thy, thz = [ np.deg2rad(a) for a in angles_deg_xyz ]
    sx, sy, sz = signs_xyz

    Rx = rotation_matrix_x(sx * thx)
    Ry = rotation_matrix_y(sy * thy)
    Rz = rotation_matrix_z(sz * thz)

    R = Rz @ Ry @ Rx
    return R @ vec


### 2. 斜めセルの構築 (例: a^0a^0c^-のための傾斜セル)

def build_tilted_supercell_cart(nx, ny, nz, a, angles_deg_xyz=(0.0,0.0,10.0), glazer_signs_xyz=(0,0,-1)):
    """
    立方晶 ABX3 を前提にしつつ、Glazer tilt による斜めセルを構築する。
    """
    e1 = np.array([1., 0., 0.])
    e2 = np.array([0., 1., 0.])
    e3 = np.array([0., 0., 1.])

    e1_t = apply_glazer_tilt(e1, angles_deg_xyz, glazer_signs_xyz)
    e2_t = apply_glazer_tilt(e2, angles_deg_xyz, glazer_signs_xyz)
    e3_t = apply_glazer_tilt(e3, angles_deg_xyz, glazer_signs_xyz)

    a_vec = a * e1_t * nx
    b_vec = a * e2_t * ny
    c_vec = a * e3_t * nz

    cell = np.array([a_vec, b_vec, c_vec])
    return cell


### 3. 格子上の A/B/X サイトの分率座標

def build_fractional_ABX3_sites(nx, ny, nz):
    """
    単位格子 a×a×a を nx×ny×nz に拡張したときの
    A/B/X サイト（分率座標）を返す。
    """
    A_sites = []
    B_sites = []
    X_sites = []

    for ix, iy, iz in product(range(nx), range(ny), range(nz)):
        shift = np.array([ix, iy, iz], dtype=float)

        # A サイト: 立方体の頂点
        for vx, vy, vz in product([0,1], repeat=3):
            A_sites.append((shift + np.array([vx, vy, vz])) / np.array([nx, ny, nz], float))

        # B サイト: 体心
        B_sites.append((shift + np.array([0.5, 0.5, 0.5])) / np.array([nx, ny, nz], float))

        # X サイト: 面心 (3種類の面)
        faces = [
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5])
        ]
        for f in faces:
            X_sites.append((shift + f) / np.array([nx, ny, nz], float))

    return np.array(A_sites), np.array(B_sites), np.array(X_sites)


### 4. 原子種の候補とインデックス

A_species_list = ["Cs", "MA", "FA"]
B_species_list = ["Ge", "Sn", "Pb"]
X_species_list = ["Cl", "Br", "I"]

A_species_to_index = {s: i for i, s in enumerate(A_species_list)}
B_species_to_index = {s: i for i, s in enumerate(B_species_list)}
X_species_to_index = {s: i for i, s in enumerate(X_species_list)}


### 5. ペアポテンシャルの読み込み

def load_potential_registry(path="pair_potentials_MACE_ABX_all/potentials_20251116.pkl"):
    registry = load_registry(path)
    return registry


### 6. AX/BX/AB 相互作用エネルギー

def pair_energy_AX(registry, elemA, elemX, distance):
    """
    registry に登録されたペアポテンシャルから AX のエネルギーを取得。
    MA/FA は分子スケーリング。
    """
    e = _safe_pair_energy(registry, "AX", elemA, elemX, float(distance))
    return scale_energy_molecular(elemA, elemX, e)

def pair_energy_BX(registry, elemB, elemX, distance):
    e = _safe_pair_energy(registry, "BX", elemB, elemX, float(distance))
    return e

def pair_energy_AB(registry, elemA, elemB, distance):
    e = _safe_pair_energy(registry, "AB", elemA, elemB, float(distance))
    return scale_energy_molecular(elemA, elemB, e)

def pair_energy_AA(registry, elemA1, elemA2, distance):
    e = _safe_pair_energy(registry, "AA", elemA1, elemA2, float(distance))
    return scale_energy_molecular(elemA1, elemA2, e)

def pair_energy_BB(registry, elemB1, elemB2, distance):
    e = _safe_pair_energy(registry, "BB", elemB1, elemB2, float(distance))
    return e

def pair_energy_XX(registry, elemX1, elemX2, distance):
    e = _safe_pair_energy(registry, "XX", elemX1, elemX2, float(distance))
    return e



### 7. 変数生成（Amplify）

def make_qubo_variables_for_composition(nx, ny, nz):
    """
    A/B/X サイトごとに one-hot 変数を生成する。
    """
    gen = VariableGenerator()
    NA = nx * ny * nz * 8   # A サイト数
    NB = nx * ny * nz       # B サイト数
    NX = nx * ny * nz * 3   # X サイト数

    qA = _gen_array_compat(gen, "qA", NA, len(A_species_list))
    qB = _gen_array_compat(gen, "qB", NB, len(B_species_list))
    qX = _gen_array_compat(gen, "qX", NX, len(X_species_list))

    return qA, qB, qX


### 8. エネルギー項の構築（組成 QUBO）

def build_energy_term_for_composition(
    cell, A_sites, B_sites, X_sites, qA, qB, qX, registry,
    d_override=None,
    cutoff_AA=None,
    cutoff_BB=None,
    cutoff_XX=None,
):
    """
    AX, BX, AB 相互作用に基づく組成エネルギー項を構築する。
    cell: 3x3 ベクトル（カーテシアン）
    A/B/X_sites: 分率座標
    """
    NA = len(A_sites)
    NB = len(B_sites)
    NX = len(X_sites)

    def frac_to_cart(frac):
        return frac[0]*cell[0] + frac[1]*cell[1] + frac[2]*cell[2]

    A_cart = np.array([frac_to_cart(f) for f in A_sites])
    B_cart = np.array([frac_to_cart(f) for f in B_sites])
    X_cart = np.array([frac_to_cart(f) for f in X_sites])

    # ここでは単純に全ペアを足す（本番では cutoff を入れる）
    energy = 0

    # AX
    for i in range(NA):
        for j in range(NX):
            rij = np.linalg.norm(A_cart[i] - X_cart[j])
            if d_override is not None:
                rij = d_override
            for ia, elemA in enumerate(A_species_list):
                for ix, elemX in enumerate(X_species_list):
                    e = pair_energy_AX(registry, elemA, elemX, rij)
                    energy += e * qA[i, ia] * qX[j, ix]

    # BX
    for i in range(NB):
        for j in range(NX):
            rij = np.linalg.norm(B_cart[i] - X_cart[j])
            if d_override is not None:
                rij = d_override
            for ib, elemB in enumerate(B_species_list):
                for ix, elemX in enumerate(X_species_list):
                    e = pair_energy_BX(registry, elemB, elemX, rij)
                    energy += e * qB[i, ib] * qX[j, ix]

    # AB
    for i in range(NA):
        for j in range(NB):
            rij = np.linalg.norm(A_cart[i] - B_cart[j])
            if d_override is not None:
                rij = d_override
            for ia, elemA in enumerate(A_species_list):
                for ib, elemB in enumerate(B_species_list):
                    e = pair_energy_AB(registry, elemA, elemB, rij)
                    energy += e * qA[i, ia] * qB[j, ib]


    # --- 追加: AA / BB / XX 相互作用（任意） ---
    # 計算コストを抑えるため、cutoff が指定された場合のみ加える。
    # registry にペアが無い場合はその組をスキップする（objective を壊さない）。
    def _has_pair(e1, e2):
        try:
            key = registry._pair_key(e1, e2)  # noqa
            return key in registry._pairs  # noqa
        except Exception:
            return False

    if cutoff_AA is not None:
        for i in range(len(A_sites)):
            for j in range(i + 1, len(A_sites)):
                rij = float(np.linalg.norm(A_cart[i] - A_cart[j]))
                if rij > float(cutoff_AA):
                    continue
                for ia, elemA1 in enumerate(A_species_list):
                    for ja, elemA2 in enumerate(A_species_list):
                        if not _has_pair(elemA1, elemA2):
                            continue
                        e = pair_energy_AA(registry, elemA1, elemA2, rij)
                        energy += e * qA[i, ia] * qA[j, ja]

    if cutoff_BB is not None:
        for i in range(len(B_sites)):
            for j in range(i + 1, len(B_sites)):
                rij = float(np.linalg.norm(B_cart[i] - B_cart[j]))
                if rij > float(cutoff_BB):
                    continue
                for ib, elemB1 in enumerate(B_species_list):
                    for jb, elemB2 in enumerate(B_species_list):
                        if not _has_pair(elemB1, elemB2):
                            continue
                        e = pair_energy_BB(registry, elemB1, elemB2, rij)
                        energy += e * qB[i, ib] * qB[j, jb]

    if cutoff_XX is not None:
        for i in range(len(X_sites)):
            for j in range(i + 1, len(X_sites)):
                rij = float(np.linalg.norm(X_cart[i] - X_cart[j]))
                if rij > float(cutoff_XX):
                    continue
                for ix, elemX1 in enumerate(X_species_list):
                    for jx, elemX2 in enumerate(X_species_list):
                        if not _has_pair(elemX1, elemX2):
                            continue
                        e = pair_energy_XX(registry, elemX1, elemX2, rij)
                        energy += e * qX[i, ix] * qX[j, jx]

    return energy


### 9. one-hot 制約と組成制約

def onehot_constraint(q, lam=10.0):
    """
    q: shape (N, K) の one-hot 変数配列
    """
    N, K = q.shape
    penalty = 0
    for i in range(N):
        penalty += lam * (sum(q[i, k] for k in range(K)) - 1) ** 2
    return penalty


def composition_constraint(q_site, target_counts, species_list, lam=1000.0):
    """
    q_site: shape (N, K)  (A/B/X いずれか)
    target_counts: {"Cs": 8, "FA":0, ...} のような dict
    species_list: そのサイトで取り得る元素名のリスト
    """
    N, K = q_site.shape
    penalty = 0
    for k, sp in enumerate(species_list):
        n_sp = sum(q_site[i, k] for i in range(N))
        target = target_counts.get(sp, 0)
        penalty += lam * (n_sp - target) ** 2
    return penalty


### 10. Fixstars クライアント

def make_fixstars_client(token=None, timeout=2000):
    if token is None:
        token = os.environ.get("FIXSTARS_TOKEN", "")
    client = FixstarsClient()
    client.token = token
    client.parameters.timeout = timeout
    return client


### 11. ABX3 組成 QUBO の構築＆解法

def build_qubo_composition_mixed_tilted_ABX_all(
    nx, ny, nz, a,
    angles_deg_xyz=(0.0, 0.0, 10.0),
    glazer_signs_xyz=(0, 0, -1),
    d_override=None,
    registry_path="pair_potentials_MACE_ABX_all/potentials_20251116.pkl",
    timeout=2000,
    cutoff_AA=None,
    cutoff_BB=None,
    cutoff_XX=None,

    target_A_counts=None,
    target_B_counts=None,
    target_X_counts=None,
    lam_onehot=10.0,
    lam_comp=10000,
    token=None,
):
    """
    ABX3 立方晶 + Glazer tilt (単一 tilt) を想定した組成 QUBO を構築し、解く。

    戻り値:
        result_comp, energy_comp, qA, qB, qX, coeffs
    """
    cell = build_tilted_supercell_cart(
        nx, ny, nz, a,
        angles_deg_xyz=angles_deg_xyz,
        glazer_signs_xyz=glazer_signs_xyz
    )

    A_sites_frac, B_sites_frac, X_sites_frac = build_fractional_ABX3_sites(nx, ny, nz)

    qA, qB, qX = make_qubo_variables_for_composition(nx, ny, nz)
    registry = load_potential_registry(registry_path)

    energy = build_energy_term_for_composition(
        cell, A_sites_frac, B_sites_frac, X_sites_frac,
        qA, qB, qX, registry,
        d_override=d_override,
        cutoff_AA=cutoff_AA,
        cutoff_BB=cutoff_BB,
        cutoff_XX=cutoff_XX,
    )

    NA = len(A_sites_frac)
    NB = len(B_sites_frac)
    NX = len(X_sites_frac)

    if target_A_counts is None:
        target_A_counts = {"Cs": 0, "MA": 0, "FA": NA}
    if target_B_counts is None:
        target_B_counts = {"Ge": 0, "Sn": 0, "Pb": NB}
    if target_X_counts is None:
        target_X_counts = {"Cl": 0, "Br": 0, "I": NX}

    energy += onehot_constraint(qA, lam=lam_onehot)
    energy += onehot_constraint(qB, lam=lam_onehot)
    energy += onehot_constraint(qX, lam=lam_onehot)

    energy += composition_constraint(qA, target_A_counts, A_species_list, lam=lam_comp)
    energy += composition_constraint(qB, target_B_counts, B_species_list, lam=lam_comp)
    energy += composition_constraint(qX, target_X_counts, X_species_list, lam=lam_comp)

    client = make_fixstars_client(token=token, timeout=timeout)
    result = solve(energy, client)

    coeffs = {
        "cell": cell,
        "coords": {"A": A_sites_frac, "B": B_sites_frac, "X": X_sites_frac},
        "A_sites": A_sites_frac,
        "B_sites": B_sites_frac,
        "X_sites": X_sites_frac,
    }

    if result is None or len(result) == 0:
        return None, energy, qA, qB, qX, coeffs

    sol = result
    chosen_A = []
    chosen_B = []
    chosen_X = []

    for i in range(NA):
        for ia, sp in enumerate(A_species_list):
            if sol.best.values[qA[i, ia]] == 1:
                chosen_A.append(sp)
                break
    for i in range(NB):
        for ib, sp in enumerate(B_species_list):
            if sol.best.values[qB[i, ib]] == 1:
                chosen_B.append(sp)
                break
    for i in range(NX):
        for ix, sp in enumerate(X_species_list):
            if sol.best.values[qX[i, ix]] == 1:
                chosen_X.append(sp)
                break

    coeffs["chosen_A"] = chosen_A
    coeffs["chosen_B"] = chosen_B
    coeffs["chosen_X"] = chosen_X

    return result, energy, qA, qB, qX, coeffs


### 12. MA/FA の配向セット（26方向）

def normalize(v):
    return v / np.linalg.norm(v)

orientation_set_26 = [
    normalize(np.array([ 1,  1,  1])),
    normalize(np.array([ 1,  1, -1])),
    normalize(np.array([ 1, -1,  1])),
    normalize(np.array([-1,  1,  1])),
    normalize(np.array([-1, -1,  1])),
    normalize(np.array([-1,  1, -1])),
    normalize(np.array([ 1, -1, -1])),
    normalize(np.array([-1, -1, -1])),
    normalize(np.array([ 0,  1,  1])),
    normalize(np.array([ 0,  1, -1])),
    normalize(np.array([ 0, -1,  1])),
    normalize(np.array([ 0, -1, -1])),
    normalize(np.array([ 1,  0,  1])),
    normalize(np.array([-1,  0,  1])),
    normalize(np.array([ 1,  0, -1])),
    normalize(np.array([-1,  0, -1])),
    normalize(np.array([ 1,  1,  0])),
    normalize(np.array([-1,  1,  0])),
    normalize(np.array([ 1, -1,  0])),
    normalize(np.array([-1, -1,  0])),
    normalize(np.array([ 1,  0,  0])),
    normalize(np.array([-1,  0,  0])),
    normalize(np.array([ 0,  1,  0])),
    normalize(np.array([ 0, -1,  0])),
    normalize(np.array([ 0,  0,  1])),
    normalize(np.array([ 0,  0, -1])),
]


### 13. ベクトル a を b に回す回転行列

def rotation_from_a_to_b(a, b):
    """
    3D ベクトル a を b に回転させる回転行列 (Rodrigues) を返す。
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        else:
            if abs(a[0]) < 0.9:
                tmp = np.array([1., 0., 0.])
            else:
                tmp = np.array([0., 1., 0.])
            v = np.cross(a, tmp)
            v /= np.linalg.norm(v)
            K = np.array([[    0, -v[2],  v[1]],
                          [ v[2],     0, -v[0]],
                          [-v[1],  v[0],    0]])
            return -np.eye(3) + 2*np.outer(v, v)
    v /= s
    K = np.array([[    0, -v[2],  v[1]],
                  [ v[2],     0, -v[0]],
                  [-v[1],  v[0],    0]])
    R = np.eye(3) + K + K@K*((1-c)/(s**2))
    return R


### 14. MA/FA カチオンの構造（テンプレート）

MA_template = {
    "C": np.array([0.0, 0.0, 0.0]),
    "N": np.array([0.0, 0.0, 1.0]),
    "H1": np.array([ 1.0, 0.0, 0.0]),
    "H2": np.array([-1.0, 0.0, 0.0]),
    "H3": np.array([0.0, 1.0, 0.0]),
    "H4": np.array([0.0,-1.0, 0.0]),
    "H5": np.array([0.0, 0.0, 2.0]),
    "H6": np.array([0.0, 0.0,-1.0]),
    "H7": np.array([1.0, 1.0, 0.0]),
    "H8": np.array([-1.0, 1.0, 0.0]),
}

FA_template = {
    "C1": np.array([-0.5, 0.0, 0.0]),
    "C2": np.array([ 0.5, 0.0, 0.0]),
    "N1": np.array([-1.5, 0.0, 0.0]),
    "N2": np.array([ 1.5, 0.0, 0.0]),
    "H1": np.array([-2.0, 1.0, 0.0]),
    "H2": np.array([-2.0,-1.0, 0.0]),
    "H3": np.array([ 2.0, 1.0, 0.0]),
    "H4": np.array([ 2.0,-1.0, 0.0]),
}


### 15. A サイト上に MA/FA を配置する関数

def place_molecule_on_site(origin_frac, orientation_vec, template, cell, scale=0.3):
    """
    origin_frac: A サイトの分率座標 (3,)
    orientation_vec: 分率空間での向きベクトル (3,)
    template: MA_template or FA_template
    cell: 3x3 カーテシアン
    scale: テンプレート座標のスケール
    """
    frac0 = np.array(origin_frac)
    cart0 = frac0[0]*cell[0] + frac0[1]*cell[1] + frac0[2]*cell[2]

    local_axes = np.eye(3)
    a = np.array([0., 0., 1.])
    b = orientation_vec / np.linalg.norm(orientation_vec)
    R = rotation_from_a_to_b(a, b)

    coords = {}

    for key, pos in template.items():
        pos_loc = np.array(pos, dtype=float) * scale
        pos_rot = R @ pos_loc
        pos_cart = cart0 + pos_rot
        coords[key] = pos_cart

    return coords


### 16. 配向 QUBO の構築（Aサイトの MA/FA の向き）

def make_qubo_variables_for_orientation(num_mol, num_orient):
    gen = VariableGenerator()
    qori = _gen_array_compat(gen, "qori", num_mol, num_orient)
    return qori


def build_energy_term_for_orientation(
    cell, A_sites, B_sites, X_sites,
    chosen_A, chosen_B, chosen_X,
    qori, registry,
    cutoff_AX=None,
    cutoff_AB=None,
    cutoff_AA=None,
):
    """
    MA/FA の向きに依存したエネルギー項。
    """
    def frac_to_cart(frac):
        return frac[0]*cell[0] + frac[1]*cell[1] + frac[2]*cell[2]

    A_cart = np.array([frac_to_cart(f) for f in A_sites])
    B_cart = np.array([frac_to_cart(f) for f in B_sites])
    X_cart = np.array([frac_to_cart(f) for f in X_sites])

    mol_indices = [i for i, sp in enumerate(chosen_A) if sp in ("MA", "FA")]
    num_mol = len(mol_indices)
    num_orient = len(orientation_set_26)

    energy = 0

    for idx_loc, iA in enumerate(mol_indices):
        A0 = A_cart[iA]
        spA = chosen_A[iA]

        for jB, Bpos in enumerate(B_cart):
            spB = chosen_B[jB]
            rij = np.linalg.norm(A0 - Bpos)
            if cutoff_AB is not None and rij > cutoff_AB:
                continue

            for io, orient_vec in enumerate(orientation_set_26):
                if spA == "MA":
                    template = MA_template
                else:
                    template = FA_template
                mol_coords = place_molecule_on_site(
                    A_sites[iA], orient_vec, template, cell
                )
                mol_HN = np.array([v for k, v in mol_coords.items() if k.startswith("H") or k.startswith("N")])
                d_min = np.min(np.linalg.norm(mol_HN - Bpos, axis=1))
                e = _registry_energy_compat(registry, "H", spB, d_min)
                energy += (e / 8.0) * qori[idx_loc, io]

        for jX, Xpos in enumerate(X_cart):
            spX = chosen_X[jX]
            rij = np.linalg.norm(A0 - Xpos)
            if cutoff_AX is not None and rij > cutoff_AX:
                continue

            for io, orient_vec in enumerate(orientation_set_26):
                if spA == "MA":
                    template = MA_template
                else:
                    template = FA_template
                mol_coords = place_molecule_on_site(
                    A_sites[iA], orient_vec, template, cell
                )
                mol_HN = np.array([v for k, v in mol_coords.items() if k.startswith("H") or k.startswith("N")])
                d_min = np.min(np.linalg.norm(mol_HN - Xpos, axis=1))
                e = _registry_energy_compat(registry, "H", spX, d_min)
                energy += (e / 8.0) * qori[idx_loc, io]

        for idx_loc2, jA in enumerate(mol_indices):
            if jA <= iA:
                continue
            A1 = A_cart[jA]
            spA2 = chosen_A[jA]
            rij = np.linalg.norm(A0 - A1)
            if cutoff_AA is not None and rij > cutoff_AA:
                continue

            for io1, orient_vec1 in enumerate(orientation_set_26):
                for io2, orient_vec2 in enumerate(orientation_set_26):
                    if spA == "MA":
                        template1 = MA_template
                    else:
                        template1 = FA_template
                    if spA2 == "MA":
                        template2 = MA_template
                    else:
                        template2 = FA_template

                    mol1 = place_molecule_on_site(A_sites[iA], orient_vec1, template1, cell)
                    mol2 = place_molecule_on_site(A_sites[jA], orient_vec2, template2, cell)

                    HN1 = np.array([v for k, v in mol1.items() if k.startswith("H") or k.startswith("N")])
                    HN2 = np.array([v for k, v in mol2.items() if k.startswith("H") or k.startswith("N")])
                    d_min = np.min(np.linalg.norm(
                        HN1[:, None, :] - HN2[None, :, :],
                        axis=2
                    ))
                    e = _registry_energy_compat(registry, "H", "H", d_min)
                    energy += (e / 64.0) * qori[idx_loc, io1] * qori[idx_loc2, io2]


    # --- 追加: AA / BB / XX 相互作用（任意） ---
    # 計算コストを抑えるため、cutoff が指定された場合のみ加える。
    # registry にペアが無い場合はその組をスキップする（objective を壊さない）。
    def _has_pair(e1, e2):
        try:
            key = registry._pair_key(e1, e2)  # noqa
            return key in registry._pairs  # noqa
        except Exception:
            return False

    if cutoff_AA is not None:
        for i in range(len(A_sites)):
            for j in range(i + 1, len(A_sites)):
                rij = float(np.linalg.norm(A_cart[i] - A_cart[j]))
                if rij > float(cutoff_AA):
                    continue
                for ia, elemA1 in enumerate(A_species_list):
                    for ja, elemA2 in enumerate(A_species_list):
                        if not _has_pair(elemA1, elemA2):
                            continue
                        e = pair_energy_AA(registry, elemA1, elemA2, rij)
                        energy += e * qA[i, ia] * qA[j, ja]

    if cutoff_BB is not None:
        for i in range(len(B_sites)):
            for j in range(i + 1, len(B_sites)):
                rij = float(np.linalg.norm(B_cart[i] - B_cart[j]))
                if rij > float(cutoff_BB):
                    continue
                for ib, elemB1 in enumerate(B_species_list):
                    for jb, elemB2 in enumerate(B_species_list):
                        if not _has_pair(elemB1, elemB2):
                            continue
                        e = pair_energy_BB(registry, elemB1, elemB2, rij)
                        energy += e * qB[i, ib] * qB[j, jb]

    if cutoff_XX is not None:
        for i in range(len(X_sites)):
            for j in range(i + 1, len(X_sites)):
                rij = float(np.linalg.norm(X_cart[i] - X_cart[j]))
                if rij > float(cutoff_XX):
                    continue
                for ix, elemX1 in enumerate(X_species_list):
                    for jx, elemX2 in enumerate(X_species_list):
                        if not _has_pair(elemX1, elemX2):
                            continue
                        e = pair_energy_XX(registry, elemX1, elemX2, rij)
                        energy += e * qX[i, ix] * qX[j, jx]

    return energy


def onehot_constraint_orientation(qori, lam=10.0):
    N, K = qori.shape
    penalty = 0
    for i in range(N):
        penalty += lam * (sum(qori[i, k] for k in range(K)) - 1) ** 2
    return penalty


def build_qubo_orientation_for_A_molecules(
    cell, coords,
    A_sites, B_sites, X_sites, box,
    chosen_A, chosen_B, chosen_X,
    registry_path="pair_potentials_MACE_ABX_all/potentials_20251116.pkl",
    lam_per_atom=5.0,
    timeout=2000,
    cutoff_AX=None,
    cutoff_AB=None,
    cutoff_AA=None,
    cutoff_BB=None,
    cutoff_XX=None,
    token=None,
    num_solutions=1,
):
    registry = load_potential_registry(registry_path)
    mol_indices = [i for i, sp in enumerate(chosen_A) if sp in ("MA", "FA")]
    num_mol = len(mol_indices)
    num_orient = len(orientation_set_26)

    if num_mol == 0:
        return None, None, None, {
            "chosen_orient": {},
            "O": None,
        }

    qori = make_qubo_variables_for_orientation(num_mol, num_orient)
    energy = build_energy_term_for_orientation(
        cell, A_sites, B_sites, X_sites,
        chosen_A, chosen_B, chosen_X,
        qori, registry,
        cutoff_AX=cutoff_AX,
        cutoff_AB=cutoff_AB,
        cutoff_AA=cutoff_AA,
    )
    energy *= lam_per_atom
    energy += onehot_constraint_orientation(qori, lam=10.0)

    client = make_fixstars_client(token=token, timeout=timeout)
    result = solve(energy, client, num_solutions=num_solutions)

    coeffs = {
        "chosen_orient": {},
        "O": orientation_set_26,
    }

    if result is None or len(result) == 0:
        return None, energy, qori, coeffs

    best = result.best
    chosen_orient = {}
    for idx_loc, iA in enumerate(mol_indices):
        for io in range(num_orient):
            if best.values[qori[idx_loc, io]] == 1:
                chosen_orient[iA] = io
                break
    coeffs["chosen_orient"] = chosen_orient

    return result, energy, qori, coeffs


### 17. MA/FA を含む ABX3 のフル構造を構築

def build_ABX3_coords_with_full_MA_FA(
    cell, coords,
    chosen_A, chosen_B, chosen_X,
    chosen_orient, O
):
    A_sites_frac = coords["A"]
    B_sites_frac = coords["B"]
    X_sites_frac = coords["X"]

    species_coords = {
        "Cs": [],
        "Ge": [], "Sn": [], "Pb": [],
        "Cl": [], "Br": [], "I": [],
        "C": [], "N": [], "H": [],
    }

    def frac_to_cart(frac):
        return frac[0]*cell[0] + frac[1]*cell[1] + frac[2]*cell[2]

    for fA, spA in zip(A_sites_frac, chosen_A):
        if spA == "Cs":
            species_coords["Cs"].append(frac_to_cart(fA))
        elif spA == "MA":
            idx = chosen_orient.get(np.where(np.all(A_sites_frac == fA, axis=1))[0][0], None)
            if idx is None:
                continue
            orient_vec = O[idx]
            mol_coords = place_molecule_on_site(fA, orient_vec, MA_template, cell)
            for key, pos in mol_coords.items():
                if key.startswith("C"):
                    species_coords["C"].append(pos)
                elif key.startswith("N"):
                    species_coords["N"].append(pos)
                elif key.startswith("H"):
                    species_coords["H"].append(pos)
        elif spA == "FA":
            idx = chosen_orient.get(np.where(np.all(A_sites_frac == fA, axis=1))[0][0], None)
            if idx is None:
                continue
            orient_vec = O[idx]
            mol_coords = place_molecule_on_site(fA, orient_vec, FA_template, cell)
            for key, pos in mol_coords.items():
                if key.startswith("C"):
                    species_coords["C"].append(pos)
                elif key.startswith("N"):
                    species_coords["N"].append(pos)
                elif key.startswith("H"):
                    species_coords["H"].append(pos)

    for fB, spB in zip(B_sites_frac, chosen_B):
        species_coords[spB].append(frac_to_cart(fB))

    for fX, spX in zip(X_sites_frac, chosen_X):
        species_coords[spX].append(frac_to_cart(fX))

    for k in list(species_coords.keys()):
        species_coords[k] = np.array(species_coords[k]) if len(species_coords[k]) > 0 else np.zeros((0,3))

    return species_coords


### 18. CIF 出力（ファイル）

def write_cif_ABX3_fullA(cell, species_coords, filename="ABX3_mixed_QA_fullA_best.cif"):
    a_vec, b_vec, c_vec = cell
    a_len = np.linalg.norm(a_vec)
    b_len = np.linalg.norm(b_vec)
    c_len = np.linalg.norm(c_vec)

    def ang(v1, v2):
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    alpha = ang(b_vec, c_vec)
    beta  = ang(a_vec, c_vec)
    gamma = ang(a_vec, b_vec)

    with open(filename, "w") as f:
        f.write(f"data_ABX3_mixed_QA_fullA\n")
        f.write(f"_cell_length_a    {a_len:.6f}\n")
        f.write(f"_cell_length_b    {b_len:.6f}\n")
        f.write(f"_cell_length_c    {c_len:.6f}\n")
        f.write(f"_cell_angle_alpha {alpha:.6f}\n")
        f.write(f"_cell_angle_beta  {beta:.6f}\n")
        f.write(f"_cell_angle_gamma {gamma:.6f}\n")
        f.write("_symmetry_space_group_name_H-M    'P1'\n")
        f.write("_symmetry_Int_Tables_number       1\n\n")

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_site_occupancy\n")

        order = ["Cs", "Ge", "Sn", "Pb", "Cl", "Br", "I", "C", "N", "H"]
        inv_cell = np.linalg.inv(cell.T)

        def cart_to_frac(pos):
            return inv_cell @ pos

        for elem in order:
            arr = species_coords.get(elem, None)
            if arr is None or len(arr) == 0:
                continue
            for i, pos_cart in enumerate(arr, start=1):
                fx, fy, fz = cart_to_frac(pos_cart)
                label = f"{elem}{i}"
                f.write(f"{label:4s} {elem:2s} {fx:.6f} {fy:.6f} {fz:.6f} 1.0\n")


# ここまでが元の mace_qubo_flow.py 本体
# ------------------------------------------------------------
# 以下が、Web アプリ等から使うための追加ユーティリティ
# ------------------------------------------------------------

# ================
# 追加ユーティリティ関数群
# ================

def cif_string_ABX3_fullA(cell, species_coords, title="ABX3_mixed_QA_fullA"):
    """write_cif_ABX3_fullA と同様の情報を、ファイルではなく文字列で返す。"""
    import io
    a_vec, b_vec, c_vec = cell
    a_len = np.linalg.norm(a_vec)
    b_len = np.linalg.norm(b_vec)
    c_len = np.linalg.norm(c_vec)

    def ang(v1, v2):
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    alpha = ang(b_vec, c_vec)
    beta = ang(a_vec, c_vec)
    gamma = ang(a_vec, b_vec)

    buf = io.StringIO()
    f = buf
    f.write(f"data_{title}\n")
    f.write(f"_cell_length_a    {a_len:.6f}\n")
    f.write(f"_cell_length_b    {b_len:.6f}\n")
    f.write(f"_cell_length_c    {c_len:.6f}\n")
    f.write(f"_cell_angle_alpha {alpha:.6f}\n")
    f.write(f"_cell_angle_beta  {beta:.6f}\n")
    f.write(f"_cell_angle_gamma {gamma:.6f}\n")
    f.write("_symmetry_space_group_name_H-M    'P1'\n")
    f.write("_symmetry_Int_Tables_number       1\n\n")

    f.write("loop_\n")
    f.write("_atom_site_label\n")
    f.write("_atom_site_type_symbol\n")
    f.write("_atom_site_fract_x\n")
    f.write("_atom_site_fract_y\n")
    f.write("_atom_site_fract_z\n")
    f.write("_atom_site_occupancy\n")

    order = ["Cs", "Ge", "Sn", "Pb", "Cl", "Br", "I", "C", "N", "H"]
    for elem in order:
        arr = species_coords.get(elem, None)
        if arr is None or len(arr) == 0:
            continue
        for i, (x, y, z) in enumerate(arr, start=1):
            label = f"{elem}{i}"
            f.write(f"{label:4s} {elem:2s} {x:.6f} {y:.6f} {z:.6f} 1.0\n")

    return buf.getvalue()


def fractions_to_counts(fracs, n_site):
    """{元素: 比率} を {元素: 個数} に変換し、合計が n_site になるようにする。"""
    if not fracs:
        return {}
    items = list(fracs.items())
    elems = [e for e, _ in items]
    raw = np.array([v for _, v in items], dtype=float)
    total = raw.sum()
    if total <= 0:
        return {}
    raw = raw / total
    counts = np.round(raw * n_site).astype(int)
    diff = int(n_site - counts.sum())
    order = np.argsort(-raw)
    i = 0
    while diff != 0 and len(order) > 0:
        idx = int(order[i % len(order)])
        if diff > 0:
            counts[idx] += 1
            diff -= 1
        else:
            if counts[idx] > 0:
                counts[idx] -= 1
                diff += 1
        i += 1
    return {elem: int(c) for elem, c in zip(elems, counts) if c > 0}


def run_full_workflow(
    A_fracs,
    B_fracs,
    X_fracs,
    nx=2,
    ny=2,
    nz=2,
    a=6.3,
    angles_deg_xyz=(0.0, 0.0, 0.0),
    glazer_signs_xyz=(0, 0, -1),
    registry_path="pair_potentials_MACE_ABX_all/potentials_20251116.pkl",
    lam_onehot=10.0,
    lam_comp=10000.0,
    timeout=2000,
    do_orientation=True,
    lam_per_atom=5.0,
    cutoff_AX=None,
    cutoff_AB=None,
    cutoff_AA=None,
    cutoff_BB=None,
    cutoff_XX=None,
    token=None,
    num_solutions=1,
):
    """ABX3 の比率指定から QUBO → (配向) → 構造生成まで一気通貫で実行する高レベル関数。"""
    from collections import Counter

    NA = nx * ny * nz
    NB = nx * ny * nz
    NX = 3 * nx * ny * nz

    target_A_counts = fractions_to_counts(A_fracs, NA)
    target_B_counts = fractions_to_counts(B_fracs, NB)
    target_X_counts = fractions_to_counts(X_fracs, NX)

    if token is None:
        token = os.environ.get("FIXSTARS_TOKEN", "")

    result_comp, energy_comp, qA, qB, qX, coeffs_comp = build_qubo_composition_mixed_tilted_ABX_all(
        nx=nx, ny=ny, nz=nz, a=a,
        angles_deg_xyz=angles_deg_xyz,
        glazer_signs_xyz=glazer_signs_xyz,
        d_override=None,
        registry_path=registry_path,
        timeout=timeout,
        target_A_counts=target_A_counts,
        target_B_counts=target_B_counts,
        target_X_counts=target_X_counts,
        lam_onehot=lam_onehot,
        lam_comp=lam_comp,
        cutoff_AA=cutoff_AA,
        cutoff_BB=cutoff_BB,
        cutoff_XX=cutoff_XX,
        token=token,
    )

    cell = coeffs_comp["cell"]
    coords = coeffs_comp["coords"]
    A_sites = coeffs_comp["A_sites"]
    B_sites = coeffs_comp["B_sites"]
    X_sites = coeffs_comp["X_sites"]
    box = coeffs_comp.get("box")
    if box is None:
        box = {"a": float(a), "b": float(a), "c": float(a), "alpha": 90.0, "beta": 90.0, "gamma": 90.0}

    chosen_A = coeffs_comp["chosen_A"]
    chosen_B = coeffs_comp["chosen_B"]
    chosen_X = coeffs_comp["chosen_X"]

    if result_comp is not None:
        E_comp = float(energy_comp.evaluate(result_comp.best.values))
    else:
        E_comp = None

    E_orient = None
    chosen_orient = {}
    O = None

    if do_orientation:
        result_ori, energy_ori, qori, coeffs_ori = build_qubo_orientation_for_A_molecules(
            cell, coords,
            A_sites, B_sites, X_sites, box,
            chosen_A, chosen_B, chosen_X,
            registry_path=registry_path,
            lam_per_atom=lam_per_atom,
            timeout=timeout,
            cutoff_AX=cutoff_AX,
            cutoff_AB=cutoff_AB,
            cutoff_AA=cutoff_AA,
            token=token,
            num_solutions=num_solutions,
        )
        if result_ori is not None:
            best = result_ori.best
            E_orient = float(energy_ori.evaluate(best.values))
            chosen_orient = coeffs_ori["chosen_orient"]
            O = coeffs_ori["O"]

    if do_orientation and chosen_orient:
        species_coords = build_ABX3_coords_with_full_MA_FA(
            cell, coords,
            chosen_A, chosen_B, chosen_X,
            chosen_orient, O,
        )
    else:
        species_coords = {
            "Cs": [],
            "Ge": [], "Sn": [], "Pb": [],
            "Cl": [], "Br": [], "I": [],
            "C": [], "N": [], "H": [],
        }
        A_frac = coords["A"]
        B_frac = coords["B"]
        X_frac = coords["X"]
        for f, sp in zip(A_frac, chosen_A):
            if sp == "Cs":
                species_coords["Cs"].append(f)
        for f, sp in zip(B_frac, chosen_B):
            species_coords[sp].append(f)
        for f, sp in zip(X_frac, chosen_X):
            species_coords[sp].append(f)
        for k in list(species_coords.keys()):
            species_coords[k] = (
                np.array(species_coords[k])
                if len(species_coords[k]) > 0
                else np.zeros((0, 3))
            )

    cif_str = cif_string_ABX3_fullA(
        cell,
        species_coords,
        title="ABX3_mixed_QA_fullA_best",
    )

    summary = {
        "A_counts_target": target_A_counts,
        "B_counts_target": target_B_counts,
        "X_counts_target": target_X_counts,
        "A_counts_actual": dict(Counter(chosen_A)),
        "B_counts_actual": dict(Counter(chosen_B)),
        "X_counts_actual": dict(Counter(chosen_X)),
        "E_comp": E_comp,
        "E_orient": E_orient,
    }

    return {
        "cell": cell,
        "coords": coords,
        "chosen_A": chosen_A,
        "chosen_B": chosen_B,
        "chosen_X": chosen_X,
        "chosen_orient": chosen_orient,
        "O": O,
        "species_coords": species_coords,
        "cif_string": cif_str,
        "cif_text": cif_str,
        "summary": summary,
    }
