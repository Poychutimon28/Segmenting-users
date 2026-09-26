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

# ช่องรับข้อมูลจากผู้ใช้
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("รายได้ต่อเดือน (บาท)", min_value=1000.0, value=30000.0)
    expense = st.number_input("รายจ่ายรวมต่อเดือน (บาท)", min_value=1000.0, value=20000.0)
    savings_rate = st.slider("อัตราส่วนการออม (Savings Rate)", 0.0, 1.0, 0.2)
    transaction_count = st.slider("จำนวนครั้งการทำรายการต่อเดือน", 1, 150, 40)

with col2:
    essential_spending = st.number_input("ค่าใช้จ่ายจำเป็น (บาท)", min_value=0.0, value=12000.0)
    discretionary_spending = st.number_input("ค่าใช้จ่ายตามใจ/ไม่จำเป็น (บาท)", min_value=0.0, value=8000.0)
    investment_amount = st.number_input("เงินลงทุนต่อเดือน (บาท)", min_value=0.0, value=3000.0)

if st.button("🔍 ทำนายกลุ่มผู้ใช้ (Predict Cluster)"):
    # คำนวณตัวแปรอัตราส่วนตามสูตร Formula
    discretionary_ratio = discretionary_spending / expense if expense > 0 else 0
    essential_ratio = essential_spending / expense if expense > 0 else 0
    investment_rate = investment_amount / income if income > 0 else 0
    
    # รวมข้อมูลอินพุต
    user_data = np.array([[savings_rate, discretionary_ratio, essential_ratio, investment_rate, transaction_count]])
    
    # Normalize ข้อมูล
    user_data_scaled = scaler.transform(user_data)
    
    # ทำนายผล
    cluster_pred = kmeans.predict(user_data_scaled)[0]
    
    st.success(f"🎉 ผลการจัดกลุ่ม: คุณอยู่ใน **Cluster {cluster_pred + 1}**")
    
    # อธิบายลักษณะกลุ่ม
    if cluster_pred == 0:
        st.info("📌 **ลักษณะกลุ่ม:** กลุ่มเน้นความคุ้มค่าและความปลอดภัยทางการเงิน ไม่ชอบความเสี่ยง เน้นการออมและควบคุมค่าใช้จ่าย")
    elif cluster_pred == 1:
        st.info("📌 **ลักษณะกลุ่ม:** กลุ่มขับเคลื่อนด้วยไลฟ์สไตล์ มีความยืดหยุ่นในการใช้จ่ายเพื่อความสุขส่วนตัวสูง")
    else:
        st.info("📌 **ลักษณะกลุ่ม:** กลุ่มแอคทีฟ มีความถี่ในการทำธุรกรรมสูง มุ่งเน้นการสร้างความมั่งคั่งและการลงทุน")
