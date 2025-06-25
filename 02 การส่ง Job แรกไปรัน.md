# **คู่มือฉบับที่ 2: การส่งงานประมวลผลแรกของคุณบน LANTA**
**(Running Your First Job on LANTA)**  

## **🔹 บทนำ**
โปรแกรมที่จะรันบนเครื่อง เรามักจะเรียกว่า จ็อบ (Job) หรือ เวิร์กโหลด (Workload) เครื่องซูเปอร์คอมพิวเตอร์มีโปรแกรมติดตั้งไว้เป็นพันตัวและมีผู้ใช้ตั้งแต่ 500 - 5,000 คน ทำให้ต้องมีระบบการจัดการคิวงาน หรือ จ็อบสเก็ดดูลิง (Job Scheduling)

LANTA ใช้ระบบที่ชื่อว่า สเลิร์ม (**Slurm**) ในการจัดการคิวงาน
เราสั่งให้สเลิร์มส่งงานไปรันในซูเปอร์คอมพิวเตอร์ ได้สองแบบ คือ ให้งานเป็นชุด (แบตช์ Batch) หรือ ให้งานแบบโต้ตอบ (Interactive) งานเป็นชุดจะต้องมีไฟล์จดงาน ที่เรียกว่า จ็อบสคริปต์ (Job Script) และออกคำสั่ง สเลิร์มแบตช์ (sbatch) ส่วนงานแบบโต้ตอบใช้คำสั่ง สเลิร์มอินเตอแรก (sinteract) หรือ สเลิร์มอัลลอค (salloc) ก็ได้ การใช้ซูเปอร์คอมพิวเตอร์ส่วนใหญ่เราจะใช้แบบแบตช์ ดังนั้นก่อนที่จะสามารถรันงานได้ นักเรียนจำเป็นต้องเข้าใจ **ขั้นตอนการส่งงาน (Job Submission)** และ **คำสั่งที่สำคัญของ Slurm**.


บทเรียนนี้ เราเตรียมไฟล์ต่าง ๆ ไว้ในโฟลเดอร์ของโปรเจ็กต์ ดังนั้นในขั้นตอนแรกเราจะคัดลอกไฟล์สำหรับฝึกซ้อมจาก Project folder มาที่ Home Folder ของเรา 

---
## 2.1 คัดลอก Folder
```bash
cp -r /project/cb900907-hpctgn/training/ .
```

## **🔹 1. ตรวจสอบสถานะของระบบ (Checking Available Resources)**  
ก่อนส่งงาน นักเรียนควรเช็คว่ามีทรัพยากร (CPU, GPU, RAM) ว่างอยู่หรือไม่ โดยใช้คำสั่ง:

```bash
sinfo
```
ตัวอย่างผลลัพธ์:
```
PARTITION    AVAIL  TIMELIMIT  NODES  STATE NODELIST
cpu          up     infinite    100   idle  x3002c0s[1-100]
gpu          up     infinite     50   idle  x3002g0s[1-50]
```
- **PARTITION** = กลุ่มของเครื่องที่ใช้รันงาน  
- **NODES** = จำนวนเครื่องที่มีอยู่ในกลุ่ม  
- **STATE** = สถานะ (idle = ว่าง, allocated = มีงานรันอยู่, down = ปิดใช้งาน)  
- **NODELIST** = รายชื่อของเครื่องที่สามารถใช้ได้  

📌 *ถ้า STATE เป็น `idle` แสดงว่ามีเครื่องพร้อมให้ใช้งาน!* 🚀

---

# 2.2. โปรแกรมที่ติดตั้งใน LANTA ติดตั้งเป็นโมดูล (module) โดยใช้ lmod
## (1) เช็คว่ามีโมดูลอะไรติดตั้งและใช้งานได้บ้าง
```bash
ml av
```

## (2) เช็คโมดูลที่โหลดแล้วในขณะนี้ (module list)
```bash
ml
```
## (3) อ่านรายการโมดูลทั้งหมดและเขียนลงไฟล์
```bash
module avail 2>&1 | tee all_modules.txt
```

## (4) เรียกโมดูลมาใช้งาน
```bash
module load <ชื่อโมดูล>
```

# 2.3. การเรียกใช้คำสั่งจากโมดูลที่ติดตั้งเอง และใช้คำสั่งในโมดูลนั้น
ในแบบฝึกนี้เราจะใช้โปรแกรม Nano และ Tree เพื่ออำนวยความสะดวก ซึ่งโครงการได้ติดตั้งโมดูลไว้ที่โฟลเดอร์ของโปรเจ็กต์ ดังนั้น เราจึงต้องออกคำสั่งเพื่อขอให้ LMod อ่านโมดูลในโฟลเดอร์โปรเจ็กต์เราด้วย
```bash
module use /project/cb900907-hpctgn/modules
```
```bash
module load nano
```

```bash
module load tree
```

## **🔹 3. เตรียมไฟล์สคริปต์สำหรับส่งงาน (Creating a Job Submission Script)**
ทุกงานที่รันบน LANTA ต้องใช้ไฟล์สคริปต์ที่กำหนด **ทรัพยากรที่ต้องใช้** และ **คำสั่งในการรันโปรแกรม** ตัวอย่างเช่น เราจะสร้างไฟล์ `myjob.sh` โดยใช้คำสั่ง:

```bash
nano myjob.sh
```
จากนั้นพิมพ์เนื้อหาดังนี้:

```bash
#!/bin/bash
#SBATCH --job-name=my_first_job  # ชื่อของงาน
#SBATCH --output=output.txt      # ไฟล์เก็บผลลัพธ์
#SBATCH --error=error.txt        # ไฟล์เก็บข้อผิดพลาด
#SBATCH --time=00:02:00          # เวลารันสูงสุด (HH:MM:SS)
#SBATCH --nodes=1                # ใช้ 1 node
#SBATCH --ntasks=1               # ใช้ 1 task
#SBATCH --cpus-per-task=4        # ใช้ 4 CPU core
#SBATCH --partition=compute      # ใช้กลุ่ม CPU
#SBATCH -A cb900908              # ระบุ Account ที่จะคิดเงิน

# คำสั่งที่ต้องการให้รัน
echo "Hello, LANTA!"
hostname
sleep 30  # หยุดรอ 30 วินาทีเพื่อทดสอบการทำงาน
echo "Job Completed!"
```

💡 **คำอธิบาย**  
✅ `#SBATCH --job-name=my_first_job` → กำหนดชื่อของงาน  
✅ `#SBATCH --output=output.txt` → บันทึกผลลัพธ์ของงานลงไฟล์  
✅ `#SBATCH --error=error.txt` → บันทึกข้อผิดพลาดที่เกิดขึ้น  
✅ `#SBATCH --time=00:02:00` → ตั้งเวลารันสูงสุด 2 นาที  
✅ `#SBATCH --nodes=1` → ใช้ 1 เครื่อง  
✅ `#SBATCH --ntasks=1` → ใช้ 1 process  
✅ `#SBATCH --cpus-per-task=4` → ใช้ 4 CPU cores  
✅ `#SBATCH --partition=compute` → ใช้พาร์ทิชัน CPU  
✅ `#SBATCH -A cb900908` → ให้ไปคิดเงินที่ account cb900908

---

## **🔹 4. ส่งงานเข้า Slurm Queue (Submitting the Job)**
หลังจากสร้างไฟล์ `myjob.sh` แล้ว ให้ใช้คำสั่งนี้เพื่อส่งงานเข้าไปในคิว:

```bash
sbatch myjob.sh
```
ตัวอย่างผลลัพธ์:
```
Submitted batch job 12345
```
หมายเลข **12345** คือ Job ID ของงานที่ถูกส่งเข้าไปในคิว

---

## **🔹 5. ตรวจสอบสถานะของงาน (Checking Job Status)**
หลังจากส่งงานแล้ว นักเรียนสามารถเช็คสถานะของงานได้โดยใช้คำสั่ง:

```bash
squeue -u $USER
```
ตัวอย่างผลลัพธ์:
```
JOBID   PARTITION  NAME          USER      STATE    TIME
12345   cpu        my_first_job  hpcuser   RUNNING  00:02:15
```
- **STATE = RUNNING** → งานกำลังทำงาน  
- **STATE = PENDING** → งานรอคิว  
- **STATE = COMPLETED** → งานเสร็จแล้ว  
- **STATE = FAILED** → งานล้มเหลว  

---

## **🔹 6. ยกเลิกงาน (Cancelling a Job)**
หากต้องการยกเลิกงานก่อนเสร็จ สามารถใช้คำสั่ง:

```bash
scancel 12345
```
(เปลี่ยน **12345** เป็น Job ID ของงานที่ต้องการยกเลิก)

---

## **🔹 7. ตรวจสอบผลลัพธ์ของงาน (Checking Job Output)**
เมื่อรันงานเสร็จแล้ว ให้เปิดไฟล์ `output.txt` เพื่อตรวจสอบผลลัพธ์:

```bash
cat output.txt
```
ตัวอย่างผลลัพธ์:
```
Hello, LANTA!
x3002c0s15b0n0
Job Completed!
```
หมายความว่าระบบได้รันคำสั่งทั้งหมดในสคริปต์เรียบร้อยแล้ว 🎉

หากมีข้อผิดพลาด ให้ตรวจสอบที่ `error.txt`:

```bash
cat error.txt
```



---

## **🔹 8. ตัวอย่างงานที่ใช้โปรแกรม Python**
หากต้องการรัน Python script บน LANTA นักเรียนต้องใช้ `module load` เพื่อโหลด Python ก่อน แล้วจึงรันโค้ดของตนเอง ตัวอย่างไฟล์ `my_python_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=python_output.txt
#SBATCH --time=00:05:00
#SBATCH --partition=compute

module load python/3.8
python my_script.py
```
จากนั้นส่งงาน:

```bash
sbatch my_python_job.sh
```

---

## **🔹 8. สรุป**
✅ ตรวจสอบสถานะของระบบ (`sinfo`)  
✅ สร้างไฟล์สคริปต์ (`myjob.sh`)  
✅ ส่งงาน (`sbatch myjob.sh`)  
✅ ตรวจสอบสถานะ (`squeue -u $USER`)  
✅ ดูผลลัพธ์ (`cat output.txt`)  
✅ ยกเลิกงานหากจำเป็น (`scancel <JobID>`)  

🎯 **คำถามสำหรับนักเรียน**
1. `#SBATCH --partition=compute` หมายถึงอะไร?
2. คำสั่ง `sbatch myjob.sh` ใช้ทำอะไร?
3. จะยกเลิกงานที่กำลังรันอยู่ได้อย่างไร?

📌 *นักเรียนสามารถทดลองส่งงานและตรวจสอบผลลัพธ์ตามตัวอย่างในคู่มือนี้ และหากมีปัญหา สามารถติดต่อ ThaiSC Support ได้ที่*  
**📧 thaisc-support@nstda.or.th**  

