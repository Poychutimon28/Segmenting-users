import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Financial Behavior Clustering App", layout="centered")

st.title("📊 ระบบจัดกลุ่มพฤติกรรมทางการเงินผู้ใช้ (k-Means)")
st.write("แอปพลิเคชันจัดกลุ่มผู้ใช้ตามพฤติกรรมการออม การใช้จ่าย และการลงทุน")

# เช็กว่ามีไฟล์โมเดลหรือยัง ถ้ายังไม่มีให้รันเทรนก่อนอัตโนมัติ
if not os.path.exists('kmeans_model.pkl') or not os.path.exists('scaler.pkl'):
    st.info("กำลังประมวลผลสร้างโมเดล k-Means จากข้อมูล...")
    import train_model

# โหลด โมเดล และ Scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

st.subheader("📝 กรอกข้อมูลทางการเงินเพื่อทดสอบทำนายกลุ่ม")

# ช่องรับข้อมูลจากผู้ใช้ (ตั้งค่าเริ่มต้น value = 0 ทั้งหมด)
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("รายได้ต่อเดือน (บาท)", min_value=0.0, value=0.0, step=1000.0)
    expense = st.number_input("รายจ่ายรวมต่อเดือน (บาท)", min_value=0.0, value=0.0, step=1000.0)
    savings_rate = st.slider("อัตราส่วนการออม (Savings Rate)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    transaction_count = st.slider("จำนวนครั้งการทำรายการต่อเดือน", min_value=0, max_value=150, value=0, step=1)

with col2:
    essential_spending = st.number_input("ค่าใช้จ่ายจำเป็น (บาท)", min_value=0.0, value=0.0, step=1000.0)
    discretionary_spending = st.number_input("ค่าใช้จ่ายตามใจ/ไม่จำเป็น (บาท)", min_value=0.0, value=0.0, step=1000.0)
    investment_amount = st.number_input("เงินลงทุนต่อเดือน (บาท)", min_value=0.0, value=0.0, step=500.0)

if st.button("🔍 ทำนายกลุ่มผู้ใช้ (Predict Cluster)"):
    # ⚠️ ตรวจสอบว่าผู้ใช้กรอกข้อมูลหรือยัง หากค่ารายได้และรายจ่ายยังเป็น 0 จะขึ้นแจ้งเตือน
    if income <= 0 or expense <= 0:
        st.warning("⚠️ กรุณากรอก 'รายได้' และ 'รายจ่ายรวม' ก่อนทำการทำนายกลุ่มครับ")
    else:
        # คำนวณตัวแปรอัตราส่วนตามสูตร Formula
        discretionary_ratio = discretionary_spending / expense
        essential_ratio = essential_spending / expense
        investment_rate = investment_amount / income
        
        # รวมข้อมูลอินพุต
        user_data = np.array([[savings_rate, discretionary_ratio, essential_ratio, investment_rate, transaction_count]])
        
        # Normalize ข้อมูล
        user_data_scaled = scaler.transform(user_data)
        
        # ทำนายผล
        cluster_pred = kmeans.predict(user_data_scaled)[0]
        
        # ปรับแมปชื่อกลุ่มให้อ่านง่ายและตรงตามผลลัพธ์โมเดล
        cluster_mapping = {
            0: {"name": "Cluster 3", "desc": "📌 **ลักษณะกลุ่ม:** กลุ่มเน้นความคุ้มค่าและความปลอดภัยทางการเงิน ไม่ชอบความเสี่ยง เน้นการออมและควบคุมค่าใช้จ่าย"},
            1: {"name": "Cluster 2", "desc": "📌 **ลักษณะกลุ่ม:** กลุ่มขับเคลื่อนด้วยไลฟ์สไตล์ มีความยืดหยุ่นในการใช้จ่ายเพื่อความสุขส่วนตัวสูง"},
            2: {"name": "Cluster 1", "desc": "📌 **ลักษณะกลุ่ม:** กลุ่มแอคทีฟ มีความถี่ในการทำธุรกรรมสูง มุ่งเน้นการสร้างความมั่งคั่งและการลงทุน"}
        }
        
        res = cluster_mapping[cluster_pred]
        
        st.success(f"🎉 ผลการจัดกลุ่ม: คุณอยู่ใน **{res['name']}**")
        st.info(res['desc'])
