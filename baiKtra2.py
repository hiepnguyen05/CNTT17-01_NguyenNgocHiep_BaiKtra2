import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc File
file_path = "/CNTT17-01_NgyenNgocHiep_BaiKTra2.xlsx"
df=pd.read_excel(file_path)
# # Đọc File theo kiểu đã tóm tắt dữ liệu
# print(df.describe())
# Đọc File nguyên dạng
print(df)
#Chuyển đổi giá trị trong cột 'Số lượng' thành kiểu số
df['Số Lượng bán'] = pd.to_numeric(df['Số lượng bán'], errors='coerce')
#Tính giá trị trung bình, bỏ qua các giá trị NaN
average = df['Số lượng bán'].mean()
#Thay thế các giá trị NaN bằng giá trị trung bình
df.fillna({'Số lượng bán':average}, inplace=True)

#Chuyển đổi giá trị trong cột 'Doanh thu (VND)' thành kiểu số
df['Doanh thu (VND)'] = pd.to_numeric(df['Doanh thu (VND)'], errors='coerce')
#Tính giá trị trung bình, bỏ qua các giá trị NaN
average = df['Doanh thu (VND)'].mean()
#Thay thế các giá trị NaN bằng giá trị trung bình
df.fillna({'Doanh thu (VND)':average}, inplace=True)


#Chuyển đổi giá trị trong cột 'Tồn kho cuối mùa' thành kiểu số
df['Tồn kho cuối mùa'] = pd.to_numeric(df['Tồn kho cuối mùa'], errors='coerce')
#Tính giá trị trung bình, bỏ qua các giá trị NaN
average = df['Tồn kho cuối mùa'].mean()
#Thay thế các giá trị NaN bằng giá trị trung bình
df.fillna({'Tồn kho cuối mùa':average}, inplace=True)

print(df)


# In ra các chỉ số thống kê cơ bản
stats = df[['Số lượng bán', 'Doanh thu (VND)', 'Tồn kho cuối mùa']].describe()
print(stats)

# Tính thêm các chỉ số thống kê khác
median = df[['Số lượng bán', 'Doanh thu (VND)', 'Tồn kho cuối mùa']].median()
variance = df[['Số lượng bán', 'Doanh thu (VND)', 'Tồn kho cuối mùa']].var()

# In ra thêm trung vị và phương sai
print("\nTrung vị:")
print(median)
print("\nPhương sai:")
print(variance)

# Vẽ biểu đồ kết hợp
plt.figure(figsize=(14, 7))

# Biểu đồ cột cho Số lượng bán
plt.subplot(2, 1, 1)
sns.barplot(x='Tên sản phẩm', y='Số lượng bán', data=df, palette='viridis')
plt.title('Số lượng bán theo tên sản phẩm')
plt.xticks(rotation=45)
plt.ylabel('Số lượng bán')

# Biểu đồ phân phối cho Doanh thu
plt.subplot(2, 1, 2)
sns.histplot(df['Doanh thu (VND)'], bins=30, kde=True, color='blue')
plt.title('Phân phối doanh thu (VND)')
plt.xlabel('Doanh thu (VND)')
plt.ylabel('Tần suất')

plt.tight_layout()
plt.show()



# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(df)

# Tách biến độc lập (X) và phụ thuộc (y)
X = df["Số lượng bán"].values.reshape(-2, 1)
y = df["Doanh thu (VND)"].values

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# Dự đoán giá trị
y_pred = model.predict(X)

# Tính toán các hệ số hồi quy
beta_0 = model.intercept_  # Hệ số chặn
beta_1 = model.coef_[0]    # Hệ số dốc

# Đánh giá mô hình
r2 = r2_score(y, y_pred)  # Hệ số xác định R^2
mse = mean_squared_error(y, y_pred)  # Sai số bình phương trung bình (MSE)

# Hiển thị kết quả
print(f"Phương trình hồi quy: y = {beta_0:.2f} + {beta_1:.2f}x")
print(f"Hệ số chặn (beta_0): {beta_0}")
print(f"Hệ số dốc (beta_1): {beta_1}")
print(f"Hệ số xác định (R^2): {r2}")
print(f"Sai số bình phương trung bình (MSE): {mse}")

# Vẽ biểu đồ
plt.scatter(X, y, color="blue", label="Dữ liệu thực tế")
plt.plot(X, y_pred, color="red", label="Dự đoán (hồi quy)")
plt.title("Hồi quy tuyến tính: Số lượng bán và Tổng doanh thu")
plt.xlabel("Số Lượng bán")
plt.ylabel("Tổng Doanh Thu")
plt.legend()
plt.grid(True)
plt.show()