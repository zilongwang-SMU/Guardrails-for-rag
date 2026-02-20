# RAG Project – Driver’s Handbook Q&A

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using:

- **LangChain**
- **Chroma Vector Database**
- **Jina Embeddings API**
- **Hugging Face Transformers (local LLM)**

The system loads a PDF (Driver’s Handbook Chapter 2), splits it into chunks, embeds the text using **Jina Embeddings**, stores vectors in **ChromaDB**, retrieves relevant context, and generates answers using a **local Hugging Face model**. **It accept user questions via command line input.**

---

## 📂 Project Structure

Rag_project/
│── main.py
│── .env
│── data/
│ └── DH-Chapter2.pdf
│── chroma_db/
│── output/
│ └── results.txt

---

## ⚙️ Requirements

- Python **3.10+** (Recommended: 3.11 / 3.12 / 3.13)
- Windows / macOS / Linux
- Internet connection (for Jina embeddings & first model download)

---

## 🚀 Installation Guide

### **1️⃣ Clone the repository**

```bash
git clone https://github.com/zilongwang-SMU/Rag_project.git
cd Rag_project
```

### **2️⃣ Create virtual environment**

Windows (PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate

```

### **3️⃣ Install dependencies**

```bash
pip install -U pip
pip install langchain langchain-community langchain-text-splitters chromadb python-dotenv requests transformers torch
```

### **4️⃣ Configure API Keys**

Create a .env file in the project root:
JINA_API_KEY=your_jina_api_key_here
Get your key from:

👉 https://jina.ai/api-dashboard/key-manager

⚠️ Do NOT use quotes.

### **5️⃣ Add the PDF**

Place the required file inside /data:
data/DH-Chapter2.pdf

### **▶️ Running the Project**

```bash
python main.py
```

### **💬 How to use this program?**

Ask questions via command line input

### **📄 Output File**

Results are saved to:

```bash
output/results.txt
```

# Guardrails-for-rag
