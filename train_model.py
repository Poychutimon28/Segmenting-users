import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# 1. อ่านข้อมูล (File)
df = pd.read_csv('personal_finance_tracker_dataset.csv')

# 2. รวมข้อมูลเป็นรายคน (Group By: user_id -> Mean)
df_grouped = df.groupby('user_id').mean(numeric_only=True)

# 3. คำนวณ Feature พฤติกรรม (Formula)
df_grouped['discretionary_ratio'] = df_grouped['discretionary_spending'] / df_grouped['monthly_expense_total']
df_grouped['essential_ratio'] = df_grouped['essential_spending'] / df_grouped['monthly_expense_total']
df_grouped['investment_rate'] = df_grouped['investment_amount'] / df_grouped['monthly_income']

# 4. เลือกเฉพาะคอลัมน์ที่จะทำ Cluster (Select Columns)
features = ['savings_rate', 'discretionary_ratio', 'essential_ratio', 'investment_rate', 'transaction_count']
X = df_grouped[features]

# 5. ปรับสเกลข้อมูล (Preprocess: Normalize)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. จัดกลุ่ม k-Means (k = 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 7. บันทึกตัวปรับสเกลและตัวโมเดลเก็บไว้
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print("สร้างไฟล์ scaler.pkl และ kmeans_model.pkl สำเร็จ!")
