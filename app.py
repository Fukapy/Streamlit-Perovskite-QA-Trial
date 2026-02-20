# app.py 2026/2/3

import os
import streamlit as st
from potential_2 import load_registry
from mace_qubo_core import run_full_workflow
from lattice_provider import lattice_from_user_or_default, lattice_predict_simple_from_csv


# =========================
# 設定
# =========================

CHARGE_TABLE_DEFAULT = {
    # Aサイト
    "Cs": +1.0,
    "Rb": +1.0,
    "K": +1.0,
    "Na": +1.0,
    "MA": +1.0,
    "FA": +1.0,

    # Bサイト
    "Ge": +2.0,
    "Sn": +2.0,
    "Pb": +2.0,

    # Xサイト
    "Cl": -1.0,
    "Br": -1.0,
    "I":  -1.0,
}

ORGANIC_NAMES = {"MA", "FA"}


# =========================
# 小物関数
# =========================

def normalize_fracs(fracs: dict) -> dict:
    total = sum(float(v) for v in fracs.values())
    if total <= 0:
        return {}
    return {str(k).strip(): float(v) / total for k, v in fracs.items() if str(k).strip() and float(v) > 0}


def composition_editor(site_label: str, default: dict, max_species: int = 6) -> dict:
    st.markdown(f"### {site_label} サイト")
    st.caption("混晶の場合は複数入力し、比率を指定してください。比率は自動で正規化します。")

    n_default = max(1, len(default))
    n = st.number_input(
        f"{site_label} に含める種の数",
        min_value=1,
        max_value=max_species,
        value=n_default,
        step=1,
        key=f"{site_label}_n",
    )

    keys = list(default.keys())
    vals = list(default.values())

    raw = {}
    for i in range(int(n)):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input(
                f"{site_label} 種 {i+1}",
                value=keys[i] if i < len(keys) else "",
                key=f"{site_label}_name_{i}",
            ).strip()
        with c2:
            frac = st.number_input(
                f"比率 {i+1}",
                min_value=0.0,
                max_value=1.0,
                value=float(vals[i]) if i < len(vals) else 0.0,
                step=0.05,
                key=f"{site_label}_frac_{i}",
            )
        if name:
            raw[name] = raw.get(name, 0.0) + float(frac)

    comp = normalize_fracs(raw)
    st.caption(f"{site_label} 比率合計（正規化前）: {sum(raw.values()):.3f}")
    return comp


def avg_charge(fracs: dict, charge_table: dict) -> float:
    if not fracs:
        raise ValueError("empty composition")
    s = sum(fracs.values())
    q = 0.0
    for k, v in fracs.items():
        if k not in charge_table:
            raise KeyError(k)
        q += (v / s) * float(charge_table[k])
    return q


def check_charge_neutral(A: dict, B: dict, X: dict, charge_table: dict, tol: float = 1e-6):
    missing = []
    for site in [A, B, X]:
        for k in site.keys():
            if k not in charge_table:
                missing.append(k)
    missing = sorted(set(missing))
    if missing:
        return False, None, None, None, missing

    qA = avg_charge(A, charge_table)
    qB = avg_charge(B, charge_table)
    qX = avg_charge(X, charge_table)
    ok = abs(qA + qB + 3.0 * qX) < tol
    return ok, qA, qB, qX, []


def get_fixstars_token_from_ui_or_env() -> str | None:
    ui_token = st.session_state.get("qa_token_input", "").strip()
    if ui_token:
        return ui_token
    env_token = os.environ.get("FIXSTARS_TOKEN", "").strip()
    if env_token:
        return env_token
    return None


def registry_has_pair(registry, e1: str, e2: str) -> bool:
    try:
        key = registry._pair_key(e1, e2)  # noqa
        return key in registry._pairs  # noqa
    except Exception:
        return False


def missing_pairs_for_ABX3(registry, A: dict, B: dict, X: dict) -> list[tuple[str, str]]:
    miss = []
    A_keys = list(A.keys())
    B_keys = list(B.keys())
    X_keys = list(X.keys())

    for a in A_keys:
        for x in X_keys:
            if not registry_has_pair(registry, a, x):
                miss.append((a, x))

    for b in B_keys:
        for x in X_keys:
            if not registry_has_pair(registry, b, x):
                miss.append((b, x))

    for a in A_keys:
        for b in B_keys:
            if not registry_has_pair(registry, a, b):
                miss.append((a, b))

    miss = sorted(set(miss))
    return miss


def has_organic_on_A(A: dict) -> bool:
    return any(a in ORGANIC_NAMES for a in A.keys())


# =========================
# UI
# =========================

st.set_page_config(page_title="Perovskite ABX3 構造探索", layout="wide")
st.title("ペロブスカイト ABX3 構造探索 Web ツール")

st.sidebar.header("量子アニーリング API")

with st.sidebar.expander("トークン設定", expanded=False):
    st.caption("環境変数 FIXSTARS_TOKEN が設定されている場合は自動で使用します。未設定の場合のみ入力してください。")
    st.text_input("Fixstars Amplify トークン", type="password", key="qa_token_input")
    if st.button("入力を消去"):
        st.session_state["qa_token_input"] = ""

st.sidebar.header("計算設定")

level = st.sidebar.radio(
    "計算精度",
    ["高速", "標準", "高精度"],
    index=1,
)

if level == "高速":
    timeout = 1000
    lam_comp = 1000.0
elif level == "高精度":
    timeout = 5000
    lam_comp = 20000.0
else:
    timeout = 2000
    lam_comp = 10000.0

nx = st.sidebar.number_input("nx", min_value=1, max_value=4, value=2, step=1)
ny = st.sidebar.number_input("ny", min_value=1, max_value=4, value=2, step=1)
nz = st.sidebar.number_input("nz", min_value=1, max_value=4, value=2, step=1)

st.sidebar.header("出力")
want_cif = st.sidebar.checkbox("CIF を出力する", True)


st.subheader("1. 組成入力")

col1, col2 = st.columns([1, 1])

with col1:
    A_fracs = composition_editor("A", default={"MA": 1.0})
    B_fracs = composition_editor("B", default={"Pb": 1.0})
    X_fracs = composition_editor("X", default={"I": 1.0})

with col2:
    st.markdown("### 入力組成（正規化後）")
    st.json({"A": A_fracs, "B": B_fracs, "X": X_fracs})

    st.markdown("### 電荷テーブル")
    st.caption("未登録の種がある場合はここに追加してください。")
    charge_table = dict(CHARGE_TABLE_DEFAULT)
    extra_lines = st.text_area(
        "追加定義（1行に 1つ。例: 「GA=+1」）",
        value="",
        help="記号は任意です。ここで入力した名前と組成入力の名前が一致する必要があります。",
    ).splitlines()

    for line in extra_lines:
        t = line.strip()
        if not t:
            continue
        if "=" not in t:
            continue
        name, val = t.split("=", 1)
        name = name.strip()
        val = val.strip()
        try:
            charge_table[name] = float(val)
        except Exception:
            pass

    st.json(charge_table)


st.subheader("2. 判定")

ok_charge, qA, qB, qX, missing_charge = check_charge_neutral(A_fracs, B_fracs, X_fracs, charge_table)

if missing_charge:
    st.warning("電荷テーブルに未登録の種があります。追加定義を入れてから判定してください。")
    st.write("未登録:", missing_charge)

if qA is not None:
    st.write("平均電荷", {"qA": qA, "qB": qB, "qX": qX})
    st.write("電荷中性チェック", {"qA + qB + 3 qX": qA + qB + 3.0 * qX})

if not ok_charge:
    st.error("③ ペロブスカイト組成ではありません。電荷の制約を満たしていません。")
    st.stop()

st.success("電荷の制約は満たしています。ABX3 の候補として扱います。")

st.subheader("3. ポテンシャル対応状況")

pot_path = st.text_input(
    "ポテンシャル pkl パス",
    value="pair_potentials_MACE_ABX_all/potentials_20251116.pkl",
    help="相対パスの場合は app.py 実行ディレクトリ基準です。",
)

registry = None
registry_error = None
try:
    registry = load_registry(pot_path)
except Exception as e:
    registry_error = str(e)

if registry is None:
    st.error("ポテンシャルの読み込みに失敗しました。パスを確認してください。")
    st.write(registry_error)
    st.stop()

miss = missing_pairs_for_ABX3(registry, A_fracs, B_fracs, X_fracs)
if miss:
    st.warning("② ペロブスカイト組成ではありますが、構造計算未対応です。必要なペアポテンシャルが不足しています。")
    st.write("不足ペア一覧（順不同）")
    st.json([{"pair": [a, b]} for a, b in miss])
    st.stop()

st.success("① ペロブスカイト組成であり、構造計算可能です。")


st.subheader("4. 格子定数と探索設定")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("### 格子定数")

    a_mode = st.radio(
        "格子定数の指定方法",
        ["デフォルト固定値を使う", "固定値を入力", "推論する"],
        index=0,
    )

    a_default = 6.30

    csv_path = st.text_input(
        "格子定数参照 CSV パス",
        value="Tilley_Perovskite_Crystals.csv",
        help="簡易推論用。存在しなければデフォルト固定値になります。",
    )


    user_a = None
    if a_mode == "固定値を入力":
        user_a = st.number_input("格子定数 a [Å]", min_value=3.0, max_value=10.0, value=a_default, step=0.05)
    elif a_mode == "推論する":
        # 簡易推論: CSV 由来ルックアップ
        lat_pred = lattice_predict_simple_from_csv(
            A_fracs=A_fracs,
            B_fracs=B_fracs,
            X_fracs=X_fracs,
            csv_path=csv_path,
            a_default=a_default,
        )
        st.caption(f"推論結果: mode={lat_pred.mode}, a={lat_pred.a:.3f} Å, note={lat_pred.note}")
        user_a = None
    else:
        st.caption(f"デフォルト値 a = {a_default} Å を使用します。")
        user_a = None

    if a_mode == "推論する":
        a_use = float(lat_pred.a)
        st.caption(f"採用: mode={lat_pred.mode}, a={a_use:.3f} Å")
    else:
        lat = lattice_from_user_or_default(
            user_a=float(user_a) if user_a is not None else None,
            A_fracs=A_fracs,
            B_fracs=B_fracs,
            X_fracs=X_fracs,
            a_default=float(a_default),
            enable_lookup=False,
            lookup_table=None,
        )
        a_use = float(lat.a)
        st.caption(f"採用: mode={lat.mode}, a={a_use:.3f} Å")

with col4:
    st.markdown("### 探索内容")
    organic = has_organic_on_A(A_fracs)
    if organic:
        st.caption("Aサイトに有機種が含まれるため、配向探索が選べます。")
    else:
        st.caption("Aサイトに有機種が含まれないため、配向探索は不要です。")

    do_orientation = False
    if organic:
        do_orientation = st.checkbox("配向探索を行う", value=True)

    st.caption("配置探索は常に実行します。")

token = get_fixstars_token_from_ui_or_env()
if token is None:
    st.warning("API トークンが未設定です。環境変数 FIXSTARS_TOKEN を設定するか、サイドバーで入力してください。")
    st.stop()

run = st.button("① の条件で探索を実行")

if run:
    st.info("探索を開始します。ログはコンソールにも出力されます。")
    result = run_full_workflow(
        A_fracs=A_fracs,
        B_fracs=B_fracs,
        X_fracs=X_fracs,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        a=float(a_use),
        angles_deg_xyz=(0.0, 0.0, 0.0),
        glazer_signs_xyz=(0, 0, -1),
        lam_onehot=10.0,
        lam_comp=float(lam_comp),
        timeout=int(timeout),
        do_orientation=bool(do_orientation),
        lam_per_atom=5.0,
        cutoff_AX=None,
        cutoff_AB=None,
        cutoff_AA=None,
        num_solutions=1,
        token=token,
    )

    st.subheader("結果サマリ")
    summary = result.get("summary", {})
    st.json(summary)

    st.subheader("内部データ確認")
    species_coords = result.get("species_coords", {})
    st.write("species_coords のキー", list(species_coords.keys()))
    st.write("species_coords の原子数", {k: int(len(v)) for k, v in species_coords.items()})

    if want_cif:
        cif_text = result.get("cif_text", None)
        if cif_text is None:
            st.warning("cif_text が result に含まれていません。CIF 出力関数の返り値を result に入れるようにしてください。")
        else:
            st.download_button(
                label="CIF をダウンロード",
                data=cif_text.encode("utf-8"),
                file_name="ABX3_mixed_QA_fullA_best.cif",
                mime="chemical/x-cif",
            )
