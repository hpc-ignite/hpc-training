### **คู่มือการ SSH เข้าใช้งาน LANTA Supercomputer ด้วย RSA Key บน macOS**  
**(สำหรับนักศึกษาที่มี SSH Key อยู่แล้ว)**  

---

## **3. ตั้งค่า SSH Config (ตัวเลือกเสริมสำหรับความสะดวก)**  
หากต้องการให้เข้าสู่ระบบได้ง่ายขึ้นโดยไม่ต้องพิมพ์พารามิเตอร์ทุกครั้ง ให้แก้ไขไฟล์ `~/.ssh/config` โดยทำดังนี้  

1. เปิด **Terminal** แล้วพิมพ์คำสั่ง:
   ```sh
   nano ~/.ssh/config
   ```
2. เพิ่มบรรทัดต่อไปนี้ (แก้ไข `your_username` ให้ตรงกับของตนเอง)
   ```
   Host lanta
       HostName lanta.nstda.or.th
       User your_username
       IdentityFile ~/.ssh/id_rsa
       ServerAliveInterval 60
   ```
3. กด **Ctrl + X** → กด **Y** → กด **Enter** เพื่อบันทึกไฟล์  

---

## **4. SSH เข้า LANTA Supercomputer**  
เมื่อมีการตั้งค่าแล้ว สามารถ SSH เข้าใช้งานได้ง่ายขึ้น  

### **4.1 ใช้วิธีปกติ**
หากไม่ได้ตั้งค่าไฟล์ `~/.ssh/config` สามารถใช้คำสั่งนี้:
```sh
ssh your_username@lanta.nstda.or.th
```

### **4.2 หากตั้งค่าใน SSH Config แล้ว**
สามารถพิมพ์คำสั่งที่สั้นลงได้:
```sh
ssh lanta
```

---

## **5. แก้ปัญหาหากเข้าสู่ระบบไม่ได้**
### **ตรวจสอบว่าใช้ Key ถูกต้องหรือไม่**
พิมพ์คำสั่ง:
```sh
ssh -v lanta
```
หากเห็นข้อความที่ไม่ใช้ `id_rsa` อาจต้องระบุไฟล์ Key ด้วยตนเอง:
```sh
ssh -i ~/.ssh/id_rsa your_username@lanta.nstda.or.th
```

