# Segmenting-users (โมเดล 3 ตัวแปร, K = 3)

ระบบต้นแบบจัดกลุ่มพฤติกรรมทางการเงินด้วย k-Means บนชุดข้อมูลสังเคราะห์
Personal Finance Tracker Dataset (Kaggle) ระบบใช้ผลการจัดกลุ่มที่ส่งออกจาก Orange Data Mining 3.40.0
(`Orange Mining/โมเดล_3ตัวแปร.ows`) ซึ่งเป็นผลที่รายงานในบทความ

## โครงสร้างไฟล์ในที่เก็บนี้

ทุกเส้นทางนับจากรากของที่เก็บ (โฟลเดอร์ `Orange Mining/` อยู่ระดับเดียวกับ `train_model.py`)

| ไฟล์ | หน้าที่ |
|---|---|
| `Orange Mining/personal_finance_tracker_dataset.csv` | ข้อมูลต้นฉบับ 3,000 ระเบียน (ไม่ได้อยู่ในที่เก็บ ดาวน์โหลดจาก Kaggle ตามหัวข้อ "ข้อมูล") |
| `Orange Mining/orange_k3_export.csv` | ผลการจัดกลุ่มที่ส่งออกจาก Orange (ผลอ้างอิงของบทความ) |
| `segment.py` | ฟังก์ชันกลาง: คำนวณอัตราส่วน ตั้งชื่อกลุ่ม และจัดกลุ่มตามจุดศูนย์กลางที่ใกล้ที่สุด |
| `train_model.py` | สร้าง `model.json` จากไฟล์ผลของ Orange (จุดศูนย์กลาง = ค่าเฉลี่ยของตัวแปรมาตรฐานของสมาชิกแต่ละกลุ่ม) |
| `app.py` | เว็บแอป Streamlit อ่าน `model.json` |
| `model.json` | ค่าที่เว็บแอปใช้ (ค่าเฉลี่ย ส่วนเบี่ยงเบนมาตรฐาน และจุดศูนย์กลาง) |

## ไฟล์ในโฟลเดอร์ Orange Mining

- `โมเดล_3ตัวแปร.ows`: แบบจำลองหลัก (ชุดที่ 2) ส่งออก `orange_k3_export.csv` (ตารางที่ 3, ไฟล์ 01–03)
- `วิเคราะห์ผล.ows`: ตารางที่ 1, 4, 6, 7 และชุดที่ 3 (ไฟล์ 05–12, 14–16)
- `ทุกคอลัมน์.ows`: ชุดที่ 1 ตัวแปรเดิม 16 ตัว (ไฟล์ 13)

**หมายเหตุเรื่องคอลัมน์ Silhouette ใน `orange_k3_export.csv`:** คอลัมน์นี้เป็นค่าที่ widget k-Means ส่งออกมาพร้อมผลการจัดกลุ่ม
ไม่ใช่ค่าที่รายงานในบทความ ค่า Silhouette ในตารางที่ 3–4 ของบทความมาจาก `03_kmeans_silhouette_k2-6.png`
(widget k-Means) และ `05_silhouette_by_cluster.csv`, `05b_negative_silhouette.csv` (widget Silhouette Plot ใน `วิเคราะห์ผล.ows`)

## ข้อมูล

ดาวน์โหลด `personal_finance_tracker_dataset.csv` จาก
https://www.kaggle.com/datasets/khushikyad001/personal-finance-tracker-dataset
แล้ววางไว้ในโฟลเดอร์ `Orange Mining/` ก่อนเปิดไฟล์ .ows หรือรัน `train_model.py`
(ไม่ได้เผยแพร่ไฟล์ข้อมูลซ้ำในที่เก็บนี้ เพราะยังไม่ได้ยืนยันเงื่อนไขการเผยแพร่ซ้ำของ Kaggle)

## ลำดับการรัน (รันจากรากของที่เก็บ)

```bash
pip install -r requirements.txt

# 1) สร้าง model.json จากผลของ Orange
python train_model.py --data "Orange Mining/personal_finance_tracker_dataset.csv" \
                      --orange-csv "Orange Mining/orange_k3_export.csv" --out model.json

# 2) เปิดเว็บแอปในเครื่อง
streamlit run app.py
```

ต้องระบุ `--data` และ `--orange-csv` ตามตัวอย่างข้างบน เพราะค่าเริ่มต้นใน `train_model.py`
ตั้งไว้สำหรับโครงสร้างโฟลเดอร์ของรายวิชา (`../Orange Mining/`)
ต้อง commit `model.json` ขึ้น GitHub ด้วย เพราะเว็บแอปไม่ฝึกโมเดลเอง

## เวอร์ชันที่ใช้

`model.json` สร้างด้วย Python 3.13.16, numpy 2.5.3, pandas 3.0.5 และ scikit-learn 1.9.1
(บันทึกไว้ในส่วน `meta` ของ `model.json`) ส่วน `requirements.txt` ระบุเวอร์ชันขั้นต่ำ
เพื่อให้ติดตั้งบน Streamlit Community Cloud ได้ ซึ่งเว็บแอปใช้เพียงค่าที่บันทึกใน `model.json`
ผลการจัดกลุ่มจึงไม่ขึ้นกับเวอร์ชันของไลบรารี

## หมายเหตุ

* ตัวเลขทุกตัวในบทความมาจากไฟล์ที่ส่งออกจาก Orange ในโฟลเดอร์ `Orange Mining/` เท่านั้น
* Orange กำหนดค่าเริ่มต้นของตัวสุ่มไม่ได้ ห้ามรัน k-Means ใหม่แล้วบันทึกทับ `orange_k3_export.csv`
* ผู้ใช้กรอกจำนวนเงินด้วยหน่วยใดก็ได้ แต่ต้องเป็นหน่วยเดียวกันทุกช่อง เพราะระบบใช้เฉพาะอัตราส่วน
  (ชุดข้อมูลไม่ระบุสกุลเงิน)
* ข้อมูลเป็นข้อมูลสังเคราะห์ ผลใช้สาธิตเชิงวิธีวิทยา ไม่ใช่คำแนะนำทางการเงิน
