"""segment.py
ฟังก์ชันกลางที่ใช้ร่วมกันระหว่าง train_model.py และ app.py

ตัวแปรที่ใช้จัดกลุ่ม (3 ตัว) นิยามตามไฟล์ Orange:
    discretionary_ratio = discretionary_spending / monthly_expense_total
    essential_ratio     = essential_spending     / monthly_expense_total
    investment_rate     = investment_amount      / monthly_income
(คำนวณจากค่าเฉลี่ยรายผู้ใช้ ตามขั้น Group by user_id = Mean ใน Orange)
"""
import json

import numpy as np

FEATURES = ["discretionary_ratio", "essential_ratio", "investment_rate"]


def compute_features(income, expense, essential, discretionary, investment):
    """คำนวณอัตราส่วน 3 ตัวจากจำนวนเงินรายเดือน (ทุกช่องต้องใช้หน่วยเงินเดียวกัน)"""
    if income is None or expense is None:
        raise ValueError("ต้องระบุรายได้และรายจ่ายรวม")
    if income <= 0 or expense <= 0:
        raise ValueError("รายได้และรายจ่ายรวมต้องมากกว่า 0")
    if min(essential, discretionary, investment) < 0:
        raise ValueError("ค่าใช้จ่ายและเงินลงทุนต้องไม่ติดลบ")
    return np.array(
        [discretionary / expense, essential / expense, investment / income],
        dtype=float,
    )


def name_clusters(centroids_z):
    """ตั้งชื่อกลุ่มจากตำแหน่งศูนย์กลาง (ไม่ผูกกับหมายเลขกลุ่มที่สลับได้)

    กฎ (ใช้กับ k = 3):
      1) กลุ่มที่ investment_rate (z) สูงสุด   -> "สัดส่วนลงทุนสูง"
      2) ในกลุ่มที่เหลือ กลุ่มที่ discretionary + essential (z) สูงกว่า
                                               -> "สัดส่วนรายจ่ายสูง"
      3) กลุ่มที่เหลือ                          -> "สัดส่วนลงทุนต่ำ รายจ่ายสมดุล"
    ชื่อเหล่านี้เป็นคำบรรยายตำแหน่งของข้อมูล ไม่ใช่การตีความเชิงพฤติกรรม
    """
    c = np.asarray(centroids_z, dtype=float)
    if c.shape != (3, 3):
        raise ValueError("name_clusters รองรับเฉพาะ k = 3 และ 3 ตัวแปร")
    names = [None] * 3
    inv = int(np.argmax(c[:, 2]))
    names[inv] = "สัดส่วนลงทุนสูง"
    rest = [i for i in range(3) if i != inv]
    spend = max(rest, key=lambda i: c[i, 0] + c[i, 1])
    names[spend] = "สัดส่วนรายจ่ายสูง"
    other = [i for i in rest if i != spend][0]
    names[other] = "สัดส่วนลงทุนต่ำ รายจ่ายสมดุล"
    return names


def load_model(path="model.json"):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["_mean"] = np.array(m["scaler"]["mean"], dtype=float)
    m["_scale"] = np.array(m["scaler"]["scale"], dtype=float)
    m["_centroids"] = np.array(m["centroids_z"], dtype=float)
    return m


def standardize(model, raw_features):
    return (np.asarray(raw_features, dtype=float) - model["_mean"]) / model["_scale"]


def predict_from_features(model, raw_features):
    """รับอัตราส่วนดิบ 3 ตัว คืนผลการจัดกลุ่มด้วย 'ศูนย์กลางที่ใกล้ที่สุด'"""
    z = standardize(model, raw_features)
    d = np.linalg.norm(model["_centroids"] - z, axis=1)
    order = np.argsort(d)
    first, second = int(order[0]), int(order[1])
    lo = np.array(model["train_range"]["p01"])
    hi = np.array(model["train_range"]["p99"])
    raw = np.asarray(raw_features, dtype=float)
    outside = [
        FEATURES[i] for i in range(len(FEATURES)) if raw[i] < lo[i] or raw[i] > hi[i]
    ]
    cl = model["clusters"][first]
    return {
        "cluster_index": first,
        "cluster_label": cl["label"],
        "cluster_name": cl["name"],
        "second_index": second,
        "second_name": model["clusters"][second]["name"],
        "distance_nearest": float(d[first]),
        "distance_second": float(d[second]),
        "margin": float(d[second] - d[first]),
        "z": z.tolist(),
        "features": raw.tolist(),
        "outside_training_range": outside,
    }


def predict(model, income, expense, essential, discretionary, investment):
    feats = compute_features(income, expense, essential, discretionary, investment)
    return predict_from_features(model, feats)
