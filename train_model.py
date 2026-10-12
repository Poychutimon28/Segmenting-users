"""train_model.py
สร้างไฟล์ model.json สำหรับเว็บแอป จากข้อมูลต้นฉบับ (โมเดล 3 ตัวแปร, k = 3)

สองโหมด:
  (1) --orange-csv data/orange_k3_export.csv   (แนะนำ ใช้ตอนเขียนบทความ)
      ใช้ "การแบ่งกลุ่มที่รายงานในบทความ" จาก Orange เป็นหลัก
      ศูนย์กลางกลุ่ม = ค่าเฉลี่ยของ z-score ของสมาชิกแต่ละกลุ่มตาม CSV ของ Orange
      แล้วตรวจว่า nearest-centroid ให้กลุ่มตรงกับ Orange กี่ราย
  (2) ไม่ใส่ --orange-csv
      ฝึก k-Means ด้วย scikit-learn (k=3, k-means++, n_init=10, random_state=42)
      ผลอาจต่างจาก Orange เล็กน้อย เพราะ k-Means หยุดที่คำตอบเฉพาะที่ต่างกันได้

ใช้:
    python train_model.py --data data/personal_finance_tracker_dataset.csv \
                          --orange-csv data/orange_k3_export.csv --out model.json
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from segment import FEATURES, name_clusters

SEED = 42
K = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_user_table(csv_path):
    """Group by user_id (Mean) แล้วคำนวณ 3 อัตราส่วน เหมือน workflow ใน Orange"""
    df = pd.read_csv(csv_path)
    g = df.groupby("user_id").mean(numeric_only=True)
    out = pd.DataFrame(
        {
            "discretionary_ratio": g["discretionary_spending"] / g["monthly_expense_total"],
            "essential_ratio": g["essential_spending"] / g["monthly_expense_total"],
            "investment_rate": g["investment_amount"] / g["monthly_income"],
            "monthly_income": g["monthly_income"],
            "sum_ratio": (g["essential_spending"] + g["discretionary_spending"]) / g["monthly_expense_total"],
            "n_months": df.groupby("user_id").size(),
        }
    )
    return df, out


def read_orange_export(path):
    """CSV จาก Orange มีแถวหัวตาราง 3 แถว (ชื่อ / ชนิด / บทบาท) ต้องข้ามสองแถวหลัง"""
    o = pd.read_csv(path, skiprows=[1, 2])
    need = {"user_id", "Cluster", *FEATURES}
    if not need.issubset(o.columns):
        raise SystemExit(f"CSV จาก Orange ต้องมีคอลัมน์ {sorted(need)}; พบ {list(o.columns)}")
    o["user_id"] = o["user_id"].astype(int)
    return o.set_index("user_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/personal_finance_tracker_dataset.csv")
    ap.add_argument("--orange-csv", default=None)
    ap.add_argument("--out", default="model.json")
    a = ap.parse_args()

    df, users = build_user_table(a.data)
    X_raw = users[FEATURES].values
    scaler = StandardScaler().fit(X_raw)  # ใช้ส่วนเบี่ยงเบนมาตรฐานแบบประชากร (ddof=0) เหมือน Orange
    Z = scaler.transform(X_raw)

    notes = {}
    if a.orange_csv:
        o = read_orange_export(a.orange_csv)
        if set(o.index) != set(users.index):
            raise SystemExit("user_id ใน CSV ของ Orange ไม่ตรงกับข้อมูลต้นฉบับ")
        o = o.loc[users.index]
        max_diff = float(np.abs(o[FEATURES].values - Z).max())
        if max_diff > 1e-3:
            raise SystemExit(
                f"z-score ใน CSV ของ Orange ต่างจากที่คำนวณซ้ำมากเกินไป (max diff = {max_diff:.5f}) "
                "ตรวจว่า Orange ใช้ตัวแปร 3 ตัวชุดเดียวกันและ Standardize แล้ว"
            )
        labels_txt = o["Cluster"].values
        uniq = sorted(set(labels_txt))
        if len(uniq) != K:
            raise SystemExit(f"พบกลุ่ม {uniq} ต้องมี {K} กลุ่ม")
        idx = np.array([uniq.index(x) for x in labels_txt])
        centroids = np.vstack([Z[idx == c].mean(axis=0) for c in range(K)])
        source = "orange_export"
        cluster_labels = uniq
        notes["max_abs_diff_z_vs_orange"] = max_diff
    else:
        km = KMeans(n_clusters=K, init="k-means++", n_init=10, max_iter=300, random_state=SEED).fit(Z)
        idx = km.labels_
        centroids = km.cluster_centers_
        source = "scikit-learn"
        cluster_labels = [f"K{c + 1}" for c in range(K)]

    # ตรวจ: การกำหนดกลุ่มแบบ nearest-centroid ตรงกับการแบ่งกลุ่มอ้างอิงกี่ราย
    d = np.linalg.norm(Z[:, None, :] - centroids[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    agree = int((nearest == idx).sum())
    notes["nearest_centroid_agreement"] = f"{agree}/{len(idx)}"

    d_sorted = np.sort(d, axis=1)
    margin = d_sorted[:, 1] - d_sorted[:, 0]
    MARGIN_T = 0.3
    notes["share_of_training_users_with_margin_lt_0.3"] = float((margin < MARGIN_T).mean())

    names = name_clusters(centroids)
    clusters = []
    for c in range(K):
        m = idx == c
        clusters.append(
            {
                "id": c,
                "label": cluster_labels[c],
                "name": names[c],
                "n": int(m.sum()),
                "share": float(m.mean()),
                "centroid_z": centroids[c].tolist(),
                "mean_raw": {f: float(users.loc[m, f].mean()) for f in FEATURES},
                "mean_income": float(users.loc[m, "monthly_income"].mean()),
            }
        )

    model = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_of_partition": source,
            "k": K,
            "seed_if_sklearn": SEED,
            "n_users": int(len(users)),
            "n_records": int(len(df)),
            "n_users_with_one_month": int((users["n_months"] == 1).sum()),
            "data_sha256": sha256(a.data),
            "orange_csv_sha256": sha256(a.orange_csv) if a.orange_csv else None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "data_note": "ชุดข้อมูลสังเคราะห์ (Kaggle: Personal Finance Tracker Dataset) ไม่ใช่ข้อมูลครัวเรือนจริง",
            **notes,
        },
        "features": FEATURES,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "centroids_z": centroids.tolist(),
        "clusters": clusters,
        "train_range": {
            "p01": np.percentile(X_raw, 1, axis=0).tolist(),
            "p99": np.percentile(X_raw, 99, axis=0).tolist(),
        },
        "sum_ratio_range": [
            float(np.percentile(users["sum_ratio"], 1)),
            float(np.percentile(users["sum_ratio"], 99)),
        ],
        "margin_threshold": MARGIN_T,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"บันทึก {a.out}")
    print(f"แหล่งการแบ่งกลุ่ม: {source}; ผู้ใช้ {len(users)} คน")
    print(f"nearest-centroid ตรงกับการแบ่งกลุ่มอ้างอิง: {agree}/{len(idx)}")
    for c in clusters:
        print(f"  {c['label']:>3} -> {c['name']:<28} n={c['n']:>4} ({c['share']:.1%})")
    if agree != len(idx):
        print("คำเตือน: nearest-centroid ไม่ตรงทั้งหมด ให้รายงานตัวเลขนี้ตามจริงในบทความ", file=sys.stderr)


if __name__ == "__main__":
    main()
