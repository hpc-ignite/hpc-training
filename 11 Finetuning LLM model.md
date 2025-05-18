### **📌 ไฟล์ Python และ Script ที่ต้องสร้างสำหรับ Fine-Tuning LLaMA 3.2:1B บน LANTA HPC**
---
การ Fine-Tuning **LLaMA 3.2:1B หรือ โมเดล LLM ใด ๆ บน LANTA HPC** เราจะต้องสร้าง **3 ไฟล์หลัก** ดังนี้:

1. **`finetune_llama.py`** → ไฟล์ซอร์สโค้ดภาษา Python ใช้สำหรับ Fine-Tuning โมเดล  
2. **`run_finetuning.slurm`** → ไฟล์ Job Script ใช้ส่งงานผ่าน SLURM บน LANTA  
3. **`convert_to_gguf.py`** → ไฟล์ Python ใช้สำหรับแปลงโมเดลให้ใช้งานกับ Ollama  

---

## **✅ 1. ไฟล์ `finetune_llama.py` (Fine-Tuning LLaMA)**
ไฟล์นี้เป็นสคริปต์ **หลัก** ที่ใช้สำหรับ **ฝึกโมเดล LLaMA 3.2:1B** โดยใช้ **QLoRA** เพื่อลดการใช้หน่วยความจำ VRAM

📌 **สร้างไฟล์ `finetune_llama.py`**:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# โหลดโมเดลและ Tokenizer
model_name = "meta-llama/Meta-Llama-3-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# โหลดโมเดลด้วย 8-bit เพื่อประหยัด VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=True,
    device_map="auto"
)

# ตั้งค่า LoRA Configuration
lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,  
    bias="none"
)
model = get_peft_model(model, lora_config)

# โหลด Dataset สำหรับ Fine-Tuning
dataset = load_dataset("json", data_files="hpc_ignite_dataset.jsonl")

# ตั้งค่าการฝึกโมเดล
training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  
    optim="adamw_torch"
)

# เริ่มการฝึกโมเดล
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save_pretrained("llama3-finetuned")
tokenizer.save_pretrained("llama3-finetuned")
```
🔹 **ไฟล์นี้ทำหน้าที่อะไร?**
- โหลดโมเดล LLaMA 3.2:1B จาก **Hugging Face**
- โหลด **Dataset JSONL** สำหรับฝึก
- ใช้ **QLoRA** เพื่อลดการใช้ VRAM
- บันทึกโมเดลที่ผ่านการฝึกแล้วไปที่ `llama3-finetuned/`

---

## **✅ 2. ไฟล์ `run_finetuning.slurm` (สคริปต์ SLURM สำหรับรัน Fine-Tuning)**
LANTA ใช้ **SLURM** สำหรับการส่งงาน **(batch job scheduling)** ดังนั้นต้องสร้างไฟล์ `.slurm` เพื่อรันงานบน **A100 GPU** 

📌 **สร้างไฟล์ `run_finetuning.slurm`**:
```bash
#!/bin/bash
#SBATCH --job-name=llama-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=llama_finetune.log

echo "🔹 โหลด Mamba Module"
module load Mamba/23.11.0-0

echo "🔹 เปิดใช้งาน Conda Environment"
conda activate /project/cb900907-hpctgn/envs/llama-finetune

echo "🔹 เริ่มกระบวนการ Fine-Tuning"
python finetune_llama.py

echo "✅ การฝึกโมเดลเสร็จสมบูรณ์"
```
🔹 **ไฟล์นี้ทำหน้าที่อะไร?**
- จอง **A100 GPU 1 ตัว**
- ใช้ **64GB RAM** และ **8 CPU Cores**
- โหลด **Mamba Module** และเปิดใช้ **Conda Environment**
- เรียกใช้ `finetune_llama.py`
- บันทึก Log ลงไฟล์ `llama_finetune.log`

📌 **รันไฟล์นี้บน LANTA โดยใช้คำสั่ง:**
```bash
sbatch run_finetuning.slurm
```

---

## **✅ 3. ไฟล์ `convert_to_gguf.py` (แปลงโมเดลให้ใช้กับ Ollama)**
หลังจาก Fine-Tuning เสร็จแล้ว ต้อง **แปลงโมเดล** ให้อยู่ใน **GGUF format** เพื่อให้ Ollama ใช้งานได้

📌 **สร้างไฟล์ `convert_to_gguf.py`**:
```python
import os

# คำสั่งแปลงโมเดลเป็น GGUF
convert_command = "python convert-hf-to-gguf.py --model_path llama3-finetuned --output llama3-finetuned.gguf"

# รันคำสั่งแปลงโมเดล
os.system(convert_command)

print("✅ แปลงโมเดลเป็น GGUF เสร็จสิ้น!")
```
🔹 **ไฟล์นี้ทำหน้าที่อะไร?**
- เรียกใช้ `convert-hf-to-gguf.py`
- แปลงโมเดลเป็น `llama3-finetuned.gguf`

📌 **รันไฟล์นี้บน LANTA:**
```bash
python convert_to_gguf.py
```

---

## **✅ 4. Deploy บน Ollama**
เมื่อแปลงโมเดลเป็น GGUF แล้ว ให้ใช้ Ollama เพื่อ Deploy

📌 **ติดตั้ง Ollama หากยังไม่มี**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

📌 **โหลดโมเดลเข้า Ollama**
```bash
ollama create hpc_ignite -f llama3-finetuned.gguf
```

📌 **ทดสอบโมเดล**
```bash
ollama run hpc_ignite "ฉันควรเริ่มเรียนรู้ HPC อย่างไร?"
```

---

## **🚀 ไฟล์ที่ต้องสร้าง**
| **ชื่อไฟล์** | **หน้าที่** |
|-------------|------------|
| **`finetune_llama.py`** | ใช้สำหรับ **ฝึกโมเดล LLaMA** ด้วย **QLoRA** |
| **`run_finetuning.slurm`** | ใช้สำหรับ **ส่งงาน SLURM บน LANTA** |
| **`convert_to_gguf.py`** | ใช้สำหรับ **แปลงโมเดลเป็น GGUF** |

---

## **📌 วิธีใช้งานทั้งหมด**
1️⃣ **อัปโหลดไฟล์ไปที่ LANTA**  
```bash
scp finetune_llama.py run_finetuning.slurm convert_to_gguf.py username@lanta.thaisc.org:~
```
2️⃣ **ล็อกอินเข้าสู่ LANTA**
```bash
ssh -X username@lanta.thaisc.org
```
3️⃣ **รัน Fine-Tuning บน SLURM**
```bash
sbatch run_finetuning.slurm
```
4️⃣ **รอการฝึกเสร็จ แล้วแปลงเป็น GGUF**
```bash
python convert_to_gguf.py
```
5️⃣ **โหลดโมเดลเข้า Ollama**
```bash
ollama create hpc_ignite -f llama3-finetuned.gguf
```
6️⃣ **ทดสอบ Agentic AI**
```bash
ollama run hpc_ignite "ฉันควรเริ่มเรียนรู้ HPC อย่างไร?"
```

---

## ** การปรับแต่งโมเดลภาษาธรรมชาติบน GPU Nodes ด้วย LLaMa-Factory**
**1. เข้าไปยังโฟลเดอร์ llm-finetuning**
```bash
cd training/llm-finetuning
```
**2. ส่ง job script ไปรันบน LANTA**
```bash 
sbatch host.sh
```
**3. เช็คสถานะงาน**
```bash 
myqueue
```
**4. อ่านไฟล์ log จากการรัน job**
```bash 
cat llama_webui_*.log
```
**5. เปิด terminal ใหม่ และ copy คำสั่งที่อ่านไฟล์ออกมา**
**6. เปิด browser ด้วย URL ด้านล่าง เพื่อเปิด LLaMa-Factory**
```bash 
http://localhost:7860
```
