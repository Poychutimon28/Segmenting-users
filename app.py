import os

import streamlit as st

from segment import FEATURES, load_model, predict

st.set_page_config(page_title="ระบบจัดกลุ่มพฤติกรรมทางการเงิน (ต้นแบบ)", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")

st.title("📊 ระบบจัดกลุ่มพฤติกรรมทางการเงิน (ต้นแบบ)")
st.write(
    "ระบบต้นแบบสำหรับสาธิตการนำแบบจำลอง k-Means (k = 3, 3 ตัวแปรอัตราส่วน) ไปใช้งาน "
    "กรอกข้อมูลรายเดือนเพื่อดูว่าอยู่ใกล้ศูนย์กลางของกลุ่มใดมากที่สุด"
)
st.warning(
    "แบบจำลองฝึกจากชุดข้อมูลสังเคราะห์ (Kaggle: Personal Finance Tracker Dataset) "
    "ผลลัพธ์ใช้เพื่อการสาธิตเท่านั้น ไม่ใช่คำแนะนำทางการเงินและไม่ใช่การวินิจฉัยพฤติกรรมจริง"
)

if not os.path.exists(MODEL_PATH):
    st.error("ไม่พบไฟล์ model.json กรุณารัน `python train_model.py` ก่อน")
    st.stop()


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


model = get_model()

st.subheader("📝 กรอกจำนวนเงินรายเดือน")
st.caption("ใช้หน่วยเงินเดียวกันทุกช่อง (สกุลใดก็ได้) เพราะระบบใช้เฉพาะอัตราส่วนระหว่างจำนวนเงิน")
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("รายได้ต่อเดือน", min_value=0.0, value=0.0, step=1000.0)
    expense = st.number_input("รายจ่ายรวมต่อเดือน", min_value=0.0, value=0.0, step=1000.0)
    investment = st.number_input("เงินลงทุนต่อเดือน", min_value=0.0, value=0.0, step=500.0)
with col2:
    essential = st.number_input("ค่าใช้จ่ายจำเป็น", min_value=0.0, value=0.0, step=1000.0)
    discretionary = st.number_input("ค่าใช้จ่ายไม่จำเป็น/ตามใจ", min_value=0.0, value=0.0, step=500.0)

if st.button("🔍 ทำนายกลุ่ม"):
    if income <= 0 or expense <= 0:
        st.warning("⚠️ กรุณากรอก 'รายได้' และ 'รายจ่ายรวม' ให้มากกว่า 0")
    else:
        res = predict(model, income, expense, essential, discretionary, investment)
        st.success(f"ใกล้ศูนย์กลางของกลุ่ม **{res['cluster_label']} — {res['cluster_name']}** มากที่สุด")

        dr, er, ir = res["features"]
        st.write(
            f"ตัวแปรที่คำนวณได้: สัดส่วนรายจ่ายไม่จำเป็น = {dr:.3f}, "
            f"สัดส่วนรายจ่ายจำเป็น = {er:.3f}, สัดส่วนเงินลงทุนต่อรายได้ = {ir:.3f}"
        )

        lo_s, hi_s = model["sum_ratio_range"]
        sum_ratio = (essential + discretionary) / expense
        if sum_ratio < lo_s or sum_ratio > hi_s:
            st.info(
                f"ค่าใช้จ่ายจำเป็นรวมกับไม่จำเป็นเท่ากับ {sum_ratio:.2f} เท่าของรายจ่ายรวม "
                f"ซึ่งอยู่นอกช่วงที่พบในข้อมูลฝึก (เปอร์เซ็นไทล์ที่ 1–99 = {lo_s:.2f}–{hi_s:.2f} เท่า) "
                "กรุณาตรวจค่าที่กรอก"
            )
        if res["outside_training_range"]:
            thai = {
                "discretionary_ratio": "สัดส่วนรายจ่ายไม่จำเป็น",
                "essential_ratio": "สัดส่วนรายจ่ายจำเป็น",
                "investment_rate": "สัดส่วนเงินลงทุนต่อรายได้",
            }
            st.info(
                "ค่าต่อไปนี้อยู่นอกช่วงที่พบบ่อยในข้อมูลฝึก (เปอร์เซ็นไทล์ที่ 1–99) "
                "ผลการจัดกลุ่มจึงน่าเชื่อถือน้อยลง: "
                + ", ".join(thai[f] for f in res["outside_training_range"])
            )
        if res["margin"] < model["margin_threshold"]:
            share = model["meta"]["share_of_training_users_with_margin_lt_0.3"]
            st.info(
                f"ผลอยู่ใกล้ขอบระหว่างกลุ่ม: ใกล้อันดับสองคือ {res['second_name']} "
                f"(ผลต่างของระยะห่าง = {res['margin']:.2f} หน่วยมาตรฐาน) "
                f"ผู้ใช้ในข้อมูลฝึกประมาณ {share:.0%} อยู่ในลักษณะนี้ เพราะกลุ่มต่าง ๆ ไม่ได้แยกจากกันชัดเจน"
            )

with st.expander("ลักษณะของแต่ละกลุ่มจากข้อมูลฝึก (ค่าเฉลี่ยของผู้ใช้ในกลุ่ม)"):
    rows = []
    for c in model["clusters"]:
        rows.append(
            {
                "กลุ่ม": f"{c['label']} — {c['name']}",
                "จำนวนผู้ใช้": c["n"],
                "สัดส่วนรายจ่ายไม่จำเป็น": round(c["mean_raw"]["discretionary_ratio"], 3),
                "สัดส่วนรายจ่ายจำเป็น": round(c["mean_raw"]["essential_ratio"], 3),
                "สัดส่วนเงินลงทุนต่อรายได้": round(c["mean_raw"]["investment_rate"], 3),
            }
        )
    st.dataframe(rows, hide_index=True)
    meta = model["meta"]
    st.caption(
        f"ฝึกจากผู้ใช้ {meta['n_users']} คน ({meta['n_records']} รายการ) | "
        f"แหล่งการแบ่งกลุ่ม: {meta['source_of_partition']} | scikit-learn {meta['scikit_learn']} | "
        f"ผู้ใช้ {meta['n_users_with_one_month']} คนมีข้อมูลเพียง 1 เดือน"
    )
    st.caption(
        "ชื่อกลุ่มเป็นคำบรรยายตำแหน่งของศูนย์กลางกลุ่มในพื้นที่ตัวแปร 3 ตัว "
        "ไม่ใช่การตีความบุคลิกหรือความเสี่ยงของผู้ใช้"
    )
