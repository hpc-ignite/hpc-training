# **🚀 8. คู่มือการใช้งาน LANTA Interactive Mode สำหรับครูและนักเรียน **

📌 **เป้าหมายของคู่มือนี้**  
- ใช้ **Interactive Mode บน LANTA HPC**  
- ใช้ **PyTorch, TensorFlow, NetCDF, และ Lightning** จาก **โมดูลที่มีอยู่**  
- แยก **Frontend** และ **Backend** อย่างชัดเจน  
- รัน **Deep Learning / AI Model Training**  

---

## **✅ 1. ตรวจสอบและเลือก Environment ที่มีอยู่**
ก่อนเริ่มต้น ให้ตรวจสอบว่า **LANTA มี Environment อะไรให้ใช้บ้าง**  
```bash
mamba env list
```
หรือ  
```bash
conda env list
```
📌 **ผลลัพธ์ตัวอย่าง:**  
```
# conda environments:
#
base                     /lustrefs/disk/modules/easybuild/software/Mamba/23.11.0-0
pytorch-2.2.2         *  /lustrefs/disk/modules/easybuild/software/Mamba/23.11.0-0/envs/pytorch-2.2.2
tensorflow-2.12.1        /lustrefs/disk/modules/easybuild/software/Mamba/23.11.0-0/envs/tensorflow-2.12.1
netcdf-py39              /lustrefs/disk/modules/easybuild/software/Mamba/23.11.0-0/envs/netcdf-py39
lightning-2.2.5          /lustrefs/disk/modules/easybuild/software/Mamba/23.11.0-0/envs/lightning-2.2.5
```
---

## **✅ 2. เชื่อมต่อ LANTA และโหลด Environment**
### **🎯 2.1 เชื่อมต่อ LANTA ผ่าน SSH**
```bash
ssh -X your_username@lanta.thaisc.org
```
📌 `-X` ใช้เพื่อเปิด GUI เช่น **Matplotlib, ParaView**  

### **🎯 2.2 โหลด Environment ที่ต้องการ**
#### **✅ ใช้ PyTorch**
```bash
module load Mamba/23.11.0-0
conda activate pytorch-2.2.2
```
#### **✅ ใช้ TensorFlow**
```bash
module load Mamba/23.11.0-0
conda activate tensorflow-2.12.1
```
#### **✅ ใช้ NetCDF (สำหรับงานพยากรณ์อากาศ WRF)**
```bash
module load Mamba/23.11.0-0
conda activate netcdf-py39
```

---

## **✅ 3. ขอใช้ Interactive Session (CPU/GPU)**
### **🎯 3.1 ขอใช้ CPU Interactive Mode**
```bash
sinteract -p compute -c 4 -t 1:00:00
```
### **🎯 3.2 ขอใช้ GPU Interactive Mode**
```bash
sinteract -p gpu -c 16 -G 1 -t 02:00:00
```
📌 ใช้ **GPU สำหรับ Deep Learning** เช่น PyTorch & TensorFlow  

---

## **✅ 4. ทดลองรัน Deep Learning ด้วย PyTorch**
📌 **ใช้ GPU สำหรับรันโมเดล AI**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# ตรวจสอบว่าใช้ GPU ได้หรือไม่
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# สร้างโมเดลง่าย ๆ
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# สร้างข้อมูลจำลอง
X = torch.randn(100, 10).to(device)
y = torch.randn(100, 1).to(device)

# เทรนโมเดล
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
print("Training complete!")
```
🎯 **รันโค้ดนี้ใน Interactive GPU Session (`sinteract -p gpu -c 16 -G 1 -t 02:00:00`)**  

---

## **✅ 5. ทดลองใช้ TensorFlow บน GPU**
📌 **รันโมเดล AI ด้วย TensorFlow**  
```python
import tensorflow as tf

# ตรวจสอบ GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# สร้างโมเดลง่าย ๆ
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mse')

# สร้างข้อมูลจำลอง
import numpy as np
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)

# เทรนโมเดล
model.fit(X, y, epochs=10)
```
🎯 **ใช้ `conda activate tensorflow-2.12.1` ก่อนรัน**  

---

## **✅ 6. แสดงผล Data Visualization บน LANTA**
📌 **เปิด GUI สำหรับ Matplotlib**
```bash
sinteract -X
```
📌 **โค้ดตัวอย่าง**
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label="Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```
🎯 **ต้องใช้ `-X` เพื่อรัน GUI**  

---

## **✅ 7. เปิด ParaView บน LANTA**
📌 **ใช้ ParaView สำหรับงาน Visualization 3D**
```bash
sinteract -p gpu -c 16 -G 1 -t 01:00:00 -X
module load ParaView/5.12.1
paraview
```
🎯 **ใช้ `-X` เพื่อเปิด GUI**  

---

## **✅ 8. ตรวจสอบและออกจาก Interactive Mode**
### **🎯 8.1 ดูว่างานที่รันอยู่คืออะไร**
```bash
squeue -u $USER
```
### **🎯 8.2 ออกจาก Interactive Mode**
```bash
exit
```

---

## **✅ 9. สรุป**
| **กิจกรรม** | **คำสั่งที่ใช้** |
|------------|----------------|
| ตรวจสอบ Environment ที่มีอยู่ | `mamba env list` |
| โหลด **PyTorch** | `conda activate pytorch-2.2.2` |
| โหลด **TensorFlow** | `conda activate tensorflow-2.12.1` |
| โหลด **NetCDF** | `conda activate netcdf-py39` |
| ขอ **CPU Interactive Session** | `sinteract -p compute -c 4 -t 1:00:00` |
| ขอ **GPU Interactive Session** | `sinteract -p gpu -c 16 -G 1 -t 02:00:00` |
| เช็ค **GPU บน LANTA** | `nvidia-smi` |
| รัน **Deep Learning (PyTorch)** | ใช้ GPU และ `torch.device("cuda")` |
| รัน **Deep Learning (TensorFlow)** | `tf.config.experimental.list_physical_devices('GPU')` |
| รัน **Matplotlib GUI** | `sinteract -X` และ `plt.show()` |
| เปิด **ParaView บน HPC** | `sinteract -p gpu -X` และ `paraview` |
| **ออกจาก Interactive Mode** | `exit` |
