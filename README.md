# Guardrails for RAG – Nova Scotia Driving Rules Assistant

## 📌 Overview

This project extends a Retrieval-Augmented Generation (RAG) system with **guardrails**, **prompt-injection defenses**, and **evaluation metrics** as required in Assignment 3.

The assistant answers questions **only about Nova Scotia driving rules** using a local vector database built from the Driver’s Handbook (DH-Chapter2.pdf).

---

## 🎯 Assignment Objectives

This implementation demonstrates:

- ✅ Input guardrails
- ✅ Output guardrails
- ✅ Execution limits
- ✅ Prompt injection defenses
- ✅ Evaluation metrics
- ✅ Structured test logging

---

## 🛡️ Guardrails Implemented

### **Input Guardrails**

| Guardrail           | Behavior                                           |
| ------------------- | -------------------------------------------------- |
| Query length limit  | Reject queries > 500 characters (`QUERY_TOO_LONG`) |
| Off-topic detection | Refuse unrelated queries (`OFF_TOPIC`)             |
| PII detection       | Redact phone/email/license plate (`PII_DETECTED`)  |

---

### **Output Guardrails**

| Guardrail                | Behavior                                             |
| ------------------------ | ---------------------------------------------------- |
| Low retrieval confidence | Refuse if similarity < threshold (`RETRIEVAL_EMPTY`) |
| Response length cap      | Limit answers to 500 words                           |
| Output validation        | Block prompt/system leakage                          |

---

### **Execution Limits**

| Limit       | Behavior                                          |
| ----------- | ------------------------------------------------- |
| LLM timeout | Abort generation after 30 seconds (`LLM_TIMEOUT`) |

---

## 🔐 Prompt Injection Defenses

The system protects against malicious instructions via:

1. **System Prompt Hardening**
   - Explicit behavioral constraints
   - Never reveal system prompts

2. **Input Sanitization**
   - Blocks phrases like:
     - "Ignore previous instructions"
     - "Print system prompt"

3. **Instruction–Data Separation**
   - Retrieved context wrapped in:
     ```
     <retrieved_context> ... </retrieved_context>
     ```

4. **Output Validation**
   - Blocks:
     - System prompt leakage
     - Meta-instruction responses

---

## 📊 Evaluation Metrics

### ✅ Retrieval Relevance

- Logs top similarity score per query
- Computes average similarity score

### ✅ Faithfulness Check

- LLM verifies: Is the answer supported by retrieved context? YES / NO

---

## 🧪 Test Scenarios

The system automatically evaluates:

### **Normal Queries**

- School bus rules
- Pedestrian yielding
- Emergency vehicle response

### **Prompt Injection Attacks**

- Instruction override attempts
- System prompt extraction
- Role reassignment attacks

### **Off-Topic & Edge Cases**

- Unrelated questions
- Queries with PII
- Empty query

---

## 📁 Output

Results saved to: output/results.txt

Each query logs:

- Query
- Guardrails Triggered
- Error Code
- Retrieved Chunks
- Similarity Score
- Answer
- Faithfulness Score

---

## ⚙️ Setup Instructions

### **1️⃣ Clone Repository**

```bash
git clone https://github.com/yourusername/Guardrails-for-rag.git
cd Guardrails-for-rag
```

### **2️⃣ Create Virtual Environment (Python 3.11 recommended)**

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### **3️⃣ Install Dependencies**

```bash
pip install -U pip setuptools wheel
pip install -r requirement.txt
```

If needed:

```bash
pip install torch transformers sentencepiece chromadb langchain langchain-community langchain-text-splitters python-dotenv requests numpy xxhash orjson grpcio
```

### **4️⃣ Configure Environment Variables**

Create .env file:
JINA_API_KEY=your_api_key_here

### **5️⃣ Add Source Document**

Place handbook PDF in: data/DH-Chapter2.pdf

### **▶️ Running the System**

```bash
python main.py
```
