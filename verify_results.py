"""verify_results.py
สคริปต์ตรวจสอบผลการวิเคราะห์ ให้ทำซ้ำได้และสืบย้อนได้ ทุกตัวเลขที่พิมพ์จะถูกเขียนลงไฟล์ใน --out

หลักการ: "การแบ่งกลุ่มที่รายงานในบทความ" คือ CSV ที่ส่งออกจาก Orange (--orange)
สคริปต์นี้ไม่ได้สร้างผลการแบ่งกลุ่มใหม่มาแทน แต่ตรวจสอบมัน

ใช้:
    python verify_results.py --data data/personal_finance_tracker_dataset.csv \
        --orange data/orange_k3_export.csv --model model.json --out verification_output

ค่า Silhouette ที่อ่านจาก widget k-Means ของ Orange (ใส่เองจากภาพหน้าจอ) ให้แก้ใน ORANGE_WIDGET_SIL ด้านล่าง
ไลบรารี: numpy, pandas, scipy, scikit-learn
"""
import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from segment import FEATURES, load_model, predict_from_features

SEED = 42
N_RESTART = 200
N_BOOT = 200
N_PERM = 200
K = 3

# ค่าที่อ่านจากหน้าต่าง k-Means ใน Orange (ภาพหน้าจอวันที่ 5/10/2569): k = 2..6
ORANGE_WIDGET_SIL = {2: 0.333, 3: 0.264, 4: 0.245, 5: 0.227, 6: 0.232}

# ชุดตัวแปรตามไฟล์ Orange (อ่านจากบทบาท 'attribute' ใน Select Columns ของ .ows)
SET_A_RAW16 = [
    "monthly_income", "monthly_expense_total", "budget_goal", "credit_score",
    "debt_to_income_ratio", "loan_payment", "investment_amount", "emergency_fund",
    "discretionary_spending", "essential_spending", "rent_or_mortgage",
    "financial_advice_score", "actual_savings", "subscription_services",
    "savings_rate", "transaction_count",
]
SET_B_EXTRA = ["savings_rate", "transaction_count", "subscription_services"]

EXTERNAL = [
    # (ชื่อคอลัมน์, หมายเหตุว่าเชื่อมกับตัวแปรที่ใช้จัดกลุ่มทางคณิตศาสตร์หรือไม่)
    ("monthly_income", "เชื่อมโดยตรง: เป็นตัวหารของ investment_rate"),
    ("actual_savings", "เชื่อมทางอ้อม: ขึ้นกับรายได้และรายจ่าย"),
    ("savings_rate", ""),
    ("savings_goal_met", "ใช้สัดส่วนเดือนที่ทำได้ตามเป้า (ค่าเฉลี่ยของ 0/1)"),
    ("transaction_count", ""),
    ("subscription_services", ""),
    ("credit_score", ""),
    ("debt_to_income_ratio", ""),
    ("financial_advice_score", ""),
    ("emergency_fund", ""),
]

LINES = []


def out(msg=""):
    print(msg)
    LINES.append(str(msg))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def zscore(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean(axis=0)) / a.std(axis=0, ddof=0)


def inertia(Z, lab):
    return float(sum(((Z[lab == c] - Z[lab == c].mean(0)) ** 2).sum() for c in np.unique(lab)))


def kmeans(Z, k=K, seed=SEED, n_init=10):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, max_iter=300, random_state=seed).fit(Z)


def sil_by_k(Z):
    return {k: float(silhouette_score(Z, kmeans(Z, k).labels_)) for k in range(2, 7)}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/personal_finance_tracker_dataset.csv")
    ap.add_argument("--orange", default="data/orange_k3_export.csv")
    ap.add_argument("--model", default="model.json")
    ap.add_argument("--out", default="verification_output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {}  # ผลทั้งหมดที่จะบันทึกเป็น JSON

    # ---------------------------------------------------------------- 0 environment
    out("=" * 70)
    out("0. สภาพแวดล้อมและหลักฐานไฟล์")
    out("=" * 70)
    env = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "seed": SEED, "n_restart": N_RESTART, "n_bootstrap": N_BOOT, "n_permutation": N_PERM,
        "data_sha256": sha256(a.data), "orange_sha256": sha256(a.orange),
    }
    R["environment"] = env
    for k, v in env.items():
        out(f"  {k}: {v}")

    # ---------------------------------------------------------------- 1 data
    df = pd.read_csv(a.data)
    g = df.groupby("user_id").mean(numeric_only=True)
    n_rows = df.groupby("user_id").size()
    ratios = pd.DataFrame({
        "discretionary_ratio": g.discretionary_spending / g.monthly_expense_total,
        "essential_ratio": g.essential_spending / g.monthly_expense_total,
        "investment_rate": g.investment_amount / g.monthly_income,
    })
    out("\n" + "=" * 70)
    out("1. ข้อมูลต้นฉบับ")
    out("=" * 70)
    d1 = {
        "n_records": int(len(df)), "n_users": int(len(g)),
        "users_with_1_record": int((n_rows == 1).sum()),
        "records_per_user_median": float(n_rows.median()), "records_per_user_max": int(n_rows.max()),
        "share_users_essential_ratio_gt_1": float((ratios.essential_ratio > 1).mean()),
        "essential_ratio_max": float(ratios.essential_ratio.max()),
        "mean_sum_essential_plus_discretionary_over_expense": float(
            ((g.essential_spending + g.discretionary_spending) / g.monthly_expense_total).mean()),
    }
    R["data"] = d1
    for k, v in d1.items():
        out(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    ratios.describe().T.round(4).to_csv(os.path.join(a.out, "01_feature_descriptives_raw.csv"), encoding="utf-8-sig")

    # ---------------------------------------------------------------- 2 orange export
    o = pd.read_csv(a.orange, skiprows=[1, 2])
    o["user_id"] = o["user_id"].astype(int)
    o = o.set_index("user_id")
    assert set(o.index) == set(g.index), "user_id ใน CSV ของ Orange ไม่ตรงกับข้อมูลต้นฉบับ"
    o = o.loc[g.index]
    Z = zscore(ratios.values)
    zdiff = float(np.abs(o[FEATURES].values - Z).max())
    lab_txt = o["Cluster"].values
    names = sorted(set(lab_txt))
    lab = np.array([names.index(x) for x in lab_txt])
    out("\n" + "=" * 70)
    out("2. ตรวจ CSV ที่ส่งออกจาก Orange")
    out("=" * 70)
    s_samples = silhouette_samples(Z, lab)
    d2 = {
        "n_rows": int(len(o)), "clusters": names,
        "max_abs_diff_z_vs_recomputed": zdiff,
        "silhouette_recomputed_from_orange_labels": float(s_samples.mean()),
        "orange_column_Silhouette_mean": float(o["Silhouette"].mean()),
        "corr_orange_Silhouette_column_vs_recomputed": float(np.corrcoef(s_samples, o["Silhouette"])[0, 1]),
        "inertia_orange_partition": inertia(Z, lab),
    }
    R["orange_export"] = d2
    for k, v in d2.items():
        out(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")
    out("  หมายเหตุ: คอลัมน์ 'Silhouette' ใน CSV ของ Orange อยู่คนละสเกลกับค่า Silhouette มาตรฐาน")
    out("            ห้ามนำค่าเฉลี่ยของคอลัมน์นั้นไปรายงานเป็นคะแนน Silhouette ให้ใช้ค่าที่คำนวณซ้ำ/ค่าจาก widget")

    # ---------------------------------------------------------------- 3 profile
    out("\n" + "=" * 70)
    out("3. โปรไฟล์กลุ่ม (จากการแบ่งกลุ่มของ Orange, หน่วยดั้งเดิม)")
    out("=" * 70)
    prof = ratios.copy()
    prof["monthly_income"] = g.monthly_income
    prof["cluster"] = lab_txt
    prof_tab = prof.groupby("cluster").agg(
        n=("monthly_income", "size"),
        discretionary_ratio=("discretionary_ratio", "mean"),
        essential_ratio=("essential_ratio", "mean"),
        investment_rate=("investment_rate", "mean"),
        monthly_income=("monthly_income", "mean"),
    )
    prof_tab["share"] = prof_tab["n"] / prof_tab["n"].sum()
    s_by = pd.DataFrame({"cluster": lab_txt, "sil": s_samples})
    prof_tab["silhouette_mean"] = s_by.groupby("cluster").sil.mean()
    prof_tab["share_negative_silhouette"] = s_by.groupby("cluster").sil.apply(lambda s: (s < 0).mean())
    prof_tab.round(4).to_csv(os.path.join(a.out, "03_cluster_profile_orange.csv"), encoding="utf-8-sig")
    out(prof_tab.round(4).to_string())
    R["profile"] = json.loads(prof_tab.round(6).to_json(orient="index"))

    # ---------------------------------------------------------------- 4 quality
    out("\n" + "=" * 70)
    out("4. คุณภาพการจัดกลุ่ม (3 ตัวแปร, k = 2..6)")
    out("=" * 70)
    sk = sil_by_k(Z)
    rows = []
    for k in range(2, 7):
        rows.append({"k": k, "silhouette_sklearn": sk[k], "silhouette_orange_widget": ORANGE_WIDGET_SIL.get(k),
                     "diff": sk[k] - ORANGE_WIDGET_SIL[k] if k in ORANGE_WIDGET_SIL else None})
    t4 = pd.DataFrame(rows)
    t4.round(4).to_csv(os.path.join(a.out, "04_silhouette_by_k.csv"), index=False)
    out(t4.round(4).to_string(index=False))
    km_ref = kmeans(Z)
    q = {
        "silhouette_k3_orange_partition": float(s_samples.mean()),
        "davies_bouldin_k3_orange_partition": float(davies_bouldin_score(Z, lab)),
        "calinski_harabasz_k3_orange_partition": float(calinski_harabasz_score(Z, lab)),
    }
    R["quality"] = {"silhouette_by_k_sklearn": sk, "orange_widget": ORANGE_WIDGET_SIL, **q}
    for k, v in q.items():
        out(f"  {k}: {v:.4f}")

    # ---------------------------------------------------------------- 5 restarts
    out("\n" + "=" * 70)
    out(f"5. ความไวต่อการเริ่มต้น: k-Means {N_RESTART} ครั้ง (n_init=1 ต่อครั้ง)")
    out("=" * 70)
    rest = []
    for sd in range(N_RESTART):
        km = KMeans(n_clusters=K, init="k-means++", n_init=1, random_state=sd).fit(Z)
        sizes = np.sort(np.bincount(km.labels_))
        rest.append({"seed": sd, "inertia": km.inertia_, "size_min": sizes[0], "size_mid": sizes[1],
                     "size_max": sizes[2], "ari_vs_orange": adjusted_rand_score(lab, km.labels_)})
    t5 = pd.DataFrame(rest)
    t5.to_csv(os.path.join(a.out, "05_restarts.csv"), index=False)
    good = t5[t5.inertia < t5.inertia.min() + 5]
    d5 = {
        "inertia_orange_partition": inertia(Z, lab),
        "inertia_min_over_restarts": float(t5.inertia.min()),
        "inertia_max_over_restarts": float(t5.inertia.max()),
        "n_runs_within_5_inertia_of_best": int(len(good)),
        "smallest_cluster_size_range_among_those": [int(good.size_min.min()), int(good.size_min.max())],
        "ari_vs_orange_median_among_those": float(good.ari_vs_orange.median()),
        "ari_vs_orange_min_among_those": float(good.ari_vs_orange.min()),
    }
    R["restarts"] = d5
    for k, v in d5.items():
        out(f"  {k}: {v}")
    out("  อ่านผล: ถ้า inertia ของหลายคำตอบที่ขนาดกลุ่มต่างกันใกล้กันมาก แปลว่าพื้นผิวเป้าหมายแบนราบ")
    out("         ขนาดกลุ่มที่รายงานจึงไม่ใช่ค่าเฉพาะตัว ให้ระบุว่ามาจากการรันครั้งใดของ Orange")

    # ---------------------------------------------------------------- 6 bootstrap
    out("\n" + "=" * 70)
    out(f"6. ความเสถียรด้วย bootstrap ({N_BOOT} ครั้ง): ฝึกบนตัวอย่างสุ่มซ้ำ ทำนายผู้ใช้ทั้งหมด เทียบกับกลุ่มของ Orange")
    out("=" * 70)
    rng = np.random.default_rng(SEED)
    aris = []
    for b in range(N_BOOT):
        idx = rng.choice(len(Z), len(Z), replace=True)
        kb = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=b).fit(Z[idx])
        aris.append(adjusted_rand_score(lab, kb.predict(Z)))
    aris = np.array(aris)
    d6 = {"ari_mean": float(aris.mean()), "ari_p05": float(np.percentile(aris, 5)),
          "ari_median": float(np.median(aris)), "ari_min": float(aris.min())}
    pd.DataFrame({"bootstrap": range(N_BOOT), "ari_vs_orange": aris}).to_csv(
        os.path.join(a.out, "06_bootstrap_ari.csv"), index=False)
    R["bootstrap"] = d6
    for k, v in d6.items():
        out(f"  {k}: {v:.4f}")

    # ---------------------------------------------------------------- 7 null baseline
    out("\n" + "=" * 70)
    out(f"7. ฐานเปรียบเทียบ: สลับค่าแต่ละตัวแปรแยกกัน {N_PERM} ครั้ง (คงการแจกแจงรายตัวแปร ทำลายความสัมพันธ์ระหว่างตัวแปร)")
    out("=" * 70)
    obs = float(silhouette_score(Z, kmeans(Z).labels_))
    rng = np.random.default_rng(SEED + 1)
    null = []
    for _ in range(N_PERM):
        P = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        null.append(silhouette_score(P, kmeans(P, seed=0).labels_))
    null = np.array(null)
    d7 = {"observed_silhouette_k3": obs, "null_mean": float(null.mean()), "null_p05": float(np.percentile(null, 5)),
          "null_p95": float(np.percentile(null, 95)), "null_max": float(null.max()),
          "p_value_ge_observed": float((1 + (null >= obs).sum()) / (1 + N_PERM))}
    pd.DataFrame({"null_silhouette": null}).to_csv(os.path.join(a.out, "07_null_silhouette.csv"), index=False)
    R["null_baseline"] = d7
    for k, v in d7.items():
        out(f"  {k}: {v:.4f}")
    out("  อ่านผล: ถ้า observed อยู่ในช่วงของฐานเปรียบเทียบ แปลว่า Silhouette ที่ได้อธิบายได้ด้วยรูปร่างการแจกแจงรายตัวแปร")
    out("         ไม่ใช่หลักฐานของกลุ่มธรรมชาติ ต้องเขียนผลเป็นการแบ่งส่วน (segmentation) ไม่ใช่การค้นพบกลุ่ม")

    # ---------------------------------------------------------------- 8 algorithms
    out("\n" + "=" * 70)
    out("8. เทียบอัลกอริทึม (k = 3, ข้อมูลชุดเดียวกัน)")
    out("=" * 70)
    ward = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Z)
    gmm = GaussianMixture(n_components=K, covariance_type="full", n_init=5, random_state=SEED).fit(Z).predict(Z)
    t8 = pd.DataFrame([
        {"method": "k-Means (Orange partition)", "silhouette": silhouette_score(Z, lab), "ari_vs_orange": 1.0},
        {"method": "Ward", "silhouette": silhouette_score(Z, ward), "ari_vs_orange": adjusted_rand_score(lab, ward)},
        {"method": "GMM (full covariance)", "silhouette": silhouette_score(Z, gmm), "ari_vs_orange": adjusted_rand_score(lab, gmm)},
    ])
    t8.round(4).to_csv(os.path.join(a.out, "08_algorithms.csv"), index=False)
    out(t8.round(4).to_string(index=False))
    R["algorithms"] = json.loads(t8.round(6).to_json(orient="records"))

    # ---------------------------------------------------------------- 9 variable sets
    out("\n" + "=" * 70)
    out("9. เปรียบเทียบชุดตัวแปรและนิยามตัวแปร (ส่วน ค) — ใช้ Standardize เหมือนกัน")
    out("   ข้อควรระวัง: Silhouette ระหว่างพื้นที่ตัวแปรที่มีจำนวนมิติต่างกันเทียบกันโดยตรงไม่ได้เต็มที่ ใช้เป็นข้อมูลประกอบ")
    out("=" * 70)
    variants = {}
    A = g[SET_A_RAW16].values
    variants["A: 16 ตัวแปรดิบ (ทุกคอลัมน์.ows)"] = zscore(A)
    B = np.column_stack([ratios.values, g[SET_B_EXTRA].values])
    variants["B: 6 ตัวแปร (โมเดล.ows เดิม)"] = zscore(B)
    variants["C: 3 อัตราส่วน (โมเดลที่รายงาน)"] = Z
    D = np.column_stack([g.discretionary_spending / g.monthly_income, g.essential_spending / g.monthly_income,
                         g.investment_amount / g.monthly_income])
    variants["D: 3 อัตราส่วน ตัวหารเป็นรายได้ทั้งหมด"] = zscore(D)
    rec = pd.DataFrame({
        "uid": df.user_id,
        "dr": df.discretionary_spending / df.monthly_expense_total,
        "er": df.essential_spending / df.monthly_expense_total,
        "ir": df.investment_amount / df.monthly_income,
    })
    E = rec.groupby("uid").mean().loc[g.index].values
    variants["E: คำนวณอัตราส่วนรายเดือนก่อน แล้วเฉลี่ยรายผู้ใช้"] = zscore(E)
    rows = []
    for name, Zv in variants.items():
        s_k = sil_by_k(Zv)
        lv = kmeans(Zv).labels_
        rows.append({"variant": name, "n_rows": len(Zv), "n_features": Zv.shape[1],
                     **{f"sil_k{k}": s_k[k] for k in range(2, 7)},
                     "davies_bouldin_k3": davies_bouldin_score(Zv, lv),
                     "ari_vs_orange_k3": adjusted_rand_score(lab, lv)})
    Fz = zscore(rec[["dr", "er", "ir"]].values)
    s_k = sil_by_k(Fz)
    lf = kmeans(Fz).labels_
    rows.append({"variant": "F: ระดับรายการ (ไม่ Group by) 3 อัตราส่วน", "n_rows": len(Fz), "n_features": 3,
                 **{f"sil_k{k}": s_k[k] for k in range(2, 7)},
                 "davies_bouldin_k3": davies_bouldin_score(Fz, lf), "ari_vs_orange_k3": None})
    t9 = pd.DataFrame(rows)
    t9.round(4).to_csv(os.path.join(a.out, "09_variable_sets.csv"), index=False, encoding="utf-8-sig")
    out(t9.round(3).to_string(index=False))
    R["variable_sets"] = json.loads(t9.round(6).to_json(orient="records"))
    out("  (ARI ของชุด F ไม่มี เพราะหน่วยวิเคราะห์ต่างกัน: 3,000 รายการ เทียบกับ 944 ผู้ใช้)")

    # ---------------------------------------------------------------- 10 external
    out("\n" + "=" * 70)
    out("10. ตัวแปรที่ไม่ได้ใช้จัดกลุ่ม (Kruskal-Wallis + ปรับ p ด้วยวิธี Holm)")
    out("=" * 70)
    rows = []
    n = len(g)
    for col, note in EXTERNAL:
        vals = g[col].values
        groups = [vals[lab == c] for c in range(K)]
        H, p = stats.kruskal(*groups)
        eps2 = H * (n + 1) / (n ** 2 - 1)
        row = {"variable": col, "note": note, "p_raw": p, "H": H, "epsilon_sq": eps2}
        for c in range(K):
            row[f"mean_{names[c]}"] = float(groups[c].mean())
        rows.append(row)
    t10 = pd.DataFrame(rows)
    t10["p_holm"] = holm(t10.p_raw.values)
    t10.to_csv(os.path.join(a.out, "10_external_variables.csv"), index=False, encoding="utf-8-sig")
    show = t10.drop(columns=["note"]).copy()
    for c in show.columns:
        if show[c].dtype == float:
            show[c] = show[c].map(lambda v: f"{v:.4g}")
    out(show.to_string(index=False))
    R["external"] = json.loads(t10.to_json(orient="records", force_ascii=False))

    # ---------------------------------------------------------------- 11 data sensitivity
    out("\n" + "=" * 70)
    out("11. ความทนทานต่อการตัดผู้ใช้ที่มีข้อมูลน้อยเดือน (ฝึกใหม่ เทียบกับกลุ่มของ Orange บนผู้ใช้ที่เหลือ)")
    out("=" * 70)
    rows = []
    for min_rows in (1, 2, 3):
        keep = (n_rows.loc[g.index] >= min_rows).values
        Zk = zscore(ratios.values[keep])
        lk = kmeans(Zk).labels_
        rows.append({"min_records_per_user": min_rows, "n_users": int(keep.sum()),
                     "silhouette_k3": float(silhouette_score(Zk, lk)),
                     "ari_vs_orange_labels": float(adjusted_rand_score(lab[keep], lk))})
    t11 = pd.DataFrame(rows)
    t11.round(4).to_csv(os.path.join(a.out, "11_min_records_sensitivity.csv"), index=False)
    out(t11.round(4).to_string(index=False))
    R["min_records"] = json.loads(t11.round(6).to_json(orient="records"))

    # ---------------------------------------------------------------- 13 ER > 1 by cluster
    out("\n" + "=" * 70)
    out("13. ผู้ใช้ที่ essential_ratio > 1 กระจายอยู่ในกลุ่มใด")
    out("=" * 70)
    er_gt1 = (ratios.essential_ratio > 1).values
    t13 = pd.DataFrame({"cluster": lab_txt, "er_gt_1": er_gt1}).groupby("cluster").er_gt_1.agg(
        n_users="size", n_er_gt_1="sum")
    t13["share_of_cluster"] = t13.n_er_gt_1 / t13.n_users
    t13["share_of_all_er_gt_1"] = t13.n_er_gt_1 / er_gt1.sum()
    t13.round(4).to_csv(os.path.join(a.out, "13_er_gt1_by_cluster.csv"), encoding="utf-8-sig")
    out(f"  ผู้ใช้ที่ ER > 1 ทั้งหมด: {int(er_gt1.sum())} คน")
    out(t13.round(4).to_string())
    R["er_gt1_by_cluster"] = json.loads(t13.round(6).to_json(orient="index"))

    # ---------------------------------------------------------------- 12 web model
    out("\n" + "=" * 70)
    out("12. ตรวจโมเดลที่เว็บแอปใช้ (model.json) ด้วยฟังก์ชันเดียวกับที่แอปเรียก")
    out("=" * 70)
    if os.path.exists(a.model):
        m = load_model(a.model)
        pred = np.array([predict_from_features(m, ratios.loc[u].values)["cluster_label"] for u in g.index])
        agree = int((pred == lab_txt).sum())
        d12 = {"agreement_with_orange": f"{agree}/{len(lab_txt)}", "model_source": m["meta"]["source_of_partition"],
               "model_data_sha256_matches_input": m["meta"]["data_sha256"] == env["data_sha256"],
               "model_orange_sha256_matches_input": m["meta"]["orange_csv_sha256"] == env["orange_sha256"]}
        R["web_model"] = d12
        for k, v in d12.items():
            out(f"  {k}: {v}")

        # ------------------------------------------------------------ 14 near-boundary share
        out("\n" + "=" * 70)
        out("14. ผู้ใช้ที่อยู่ใกล้ขอบระหว่างกลุ่ม (เกณฑ์เดียวกับคำเตือนในเว็บแอป)")
        out("=" * 70)
        Zm = (ratios.values - m["_mean"]) / m["_scale"]
        dist = np.sort(np.linalg.norm(Zm[:, None, :] - m["_centroids"][None, :, :], axis=2), axis=1)
        margin = dist[:, 1] - dist[:, 0]
        thr = m["margin_threshold"]
        n_near = int((margin < thr).sum())
        d14 = {"threshold": thr, "n_users_near_boundary": n_near, "n_users": int(len(margin)),
               "share_near_boundary": n_near / len(margin)}
        R["near_boundary"] = d14
        for k, v in d14.items():
            out(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        out(f"  ไม่พบ {a.model} (ข้ามขั้นตอนนี้) รัน train_model.py ก่อน")

    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    out(f"\nบันทึกผลทั้งหมดใน {a.out}/ (summary.txt, results.json และไฟล์ตาราง 01-11)")


if __name__ == "__main__":
    sys.exit(main())
