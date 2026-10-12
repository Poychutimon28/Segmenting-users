# Segmenting-users (โมเดล 3 ตัวแปร, k = 3)

ระบบต้นแบบจัดกลุ่มพฤติกรรมทางการเงินด้วย k-Means บนชุดข้อมูลสังเคราะห์
Personal Finance Tracker Dataset (Kaggle) ใช้การแบ่งกลุ่มจาก Orange เป็นผลหลักที่รายงานในบทความ

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `data/personal_finance_tracker_dataset.csv` | ข้อมูลต้นฉบับ 3,000 รายการ |
| `data/orange_k3_export.csv` | ผลการแบ่งกลุ่มที่ส่งออกจาก Orange (`โมเดล_3ตัวแปร.ows`) ใช้เป็นผลอ้างอิงของบทความ |
| `segment.py` | ฟังก์ชันกลาง: คำนวณอัตราส่วน ตั้งชื่อกลุ่ม และทำนายกลุ่มแบบศูนย์กลางที่ใกล้ที่สุด |
| `train_model.py` | สร้าง `model.json` จากผลของ Orange (หรือจาก scikit-learn ถ้าไม่ระบุไฟล์ Orange) |
| `app.py` | เว็บแอป Streamlit อ่าน `model.json` |
| `verify_results.py` | สคริปต์ตรวจสอบผล เขียนผลลงโฟลเดอร์ `verification_output/` |

## ลำดับการรัน

```bash
pip install -r requirements.txt

# 1) สร้างโมเดลของเว็บจากผลของ Orange
python train_model.py --data data/personal_finance_tracker_dataset.csv \
                      --orange-csv data/orange_k3_export.csv --out model.json

# 2) ตรวจสอบผลทั้งหมด (ประมาณ 1 นาที)
python verify_results.py --data data/personal_finance_tracker_dataset.csv \
       --orange data/orange_k3_export.csv --model model.json --out verification_output

# 3) เปิดเว็บแอปในเครื่อง
streamlit run app.py
```

ต้อง commit `model.json` ขึ้น GitHub ด้วย เพราะเว็บแอปไม่ฝึกโมเดลเองแล้ว
ไฟล์ `scaler.pkl` และ `kmeans_model.pkl` เดิมไม่ใช้แล้ว ลบออกได้

## หลักการเรื่องตัวเลขในบทความ

* ผลการแบ่งกลุ่ม (ขนาดกลุ่ม ค่าเฉลี่ยต่อกลุ่ม) มาจาก `data/orange_k3_export.csv` เท่านั้น
  Orange กำหนด random seed ไม่ได้ การรันใหม่อาจได้ขนาดกลุ่มต่างไปเล็กน้อย ห้ามรันใหม่แล้วปนตัวเลขสองรอบ
* ตัวเลขตรวจสอบ (ความเสถียร ฐานเปรียบเทียบ ตัวแปรภายนอก) ให้ใช้จาก `verification_output/` ที่รันในเครื่องของผู้เขียนเอง
  และบันทึกเวอร์ชันไลบรารีกับ hash ของไฟล์ที่พิมพ์ในส่วน 0 ของ `summary.txt`
* ห้ามรายงานค่าเฉลี่ยของคอลัมน์ `Silhouette` ใน CSV ของ Orange เป็นคะแนน Silhouette (คนละสเกล)
* ข้อมูลเป็นข้อมูลสังเคราะห์ ผลใช้สรุปเชิงวิธีวิทยา ไม่ใช่พฤติกรรมครัวเรือนจริง
