# 🧠 Multimodal RAG Chat Application

## 🔗 Live Demo : https://multimodal-rag-hbqy96cmphnld5ceqtb5cv.streamlit.app/

---

## 📌 Overview

This project is a **Multimodal Retrieval-Augmented Generation (RAG) system** built using **Streamlit, LangChain, ChromaDB, and Llama models via Groq**.

It allows users to upload PDF documents and interact with them through natural language queries. The system intelligently processes **text, tables, and images**, enabling contextual question answering across multiple data modalities.

---

## 🚀 Key Features

* 📄 **PDF Understanding** — Extracts and processes document text
* 📊 **Table Extraction** — Converts tables into structured markdown
* 🖼️ **Image Analysis** — Uses vision models to generate image descriptions
* 🔍 **Semantic Search** — Retrieves relevant content using embeddings
* 💬 **Conversational Q&A** — Supports multi-turn chat with context awareness
* 🧠 **Multimodal Reasoning** — Combines text, tables, and image insights
* 📚 **Source Attribution** — Displays document sources for answers
* 💾 **Persistent Storage** — Saves embeddings using ChromaDB

---

## 🏗️ System Architecture

### 1. Data Ingestion

* PDFs uploaded via Streamlit UI
* Parsed using **PyMuPDF (fitz)**

### 2. Content Extraction

* Text → Direct extraction
* Tables → Converted to markdown
* Images → Extracted and captioned using vision model

### 3. Multimodal Processing

* Image captions generated using:

  * `llama-3.2-90b-vision-preview`
* Converted into textual representations for indexing

### 4. Chunking Strategy

* **Semantic Chunking (Primary)** using embeddings
* **Recursive Chunking (Fallback)** for robustness
* Structure-aware handling:

  * Text → chunked
  * Tables/Images → preserved as-is

### 5. Embedding Layer

* Model: `all-MiniLM-L6-v2`
* Converts chunks into dense vector representations

### 6. Vector Database

* **ChromaDB**
* Stores:

  * text chunks
  * metadata (source, page, type)
  * image references

### 7. Retrieval Mechanism

* Uses **Max Marginal Relevance (MMR)** for diverse results
* Fetches top relevant chunks

### 8. Generation Layer

* Model: `llama-3.3-70b-versatile`
* Combines:

  * retrieved context
  * user query
  * conversation history

---

## 🧩 Tech Stack

| Component      | Technology              |
| -------------- | ----------------------- |
| UI             | Streamlit               |
| LLM (Text)     | Llama 3.3 (Groq)        |
| LLM (Vision)   | Llama 3.2 Vision (Groq) |
| Embeddings     | HuggingFace (MiniLM)    |
| Vector DB      | ChromaDB                |
| PDF Processing | PyMuPDF                 |
| Framework      | LangChain               |

---

## ⚙️ What This System Can Do

✅ Answer questions from PDFs
✅ Understand charts and diagrams (via captions)
✅ Extract and use tabular data
✅ Maintain conversational context
✅ Provide source-based answers
✅ Work across multiple uploaded documents

---

## 🔮 Future Improvements

### 🚀 High Impact Upgrades

* Add **Cross-Encoder Reranking** for better accuracy
* Implement **Hybrid Search (BM25 + Vector)**
* Use **CLIP embeddings** for true multimodal retrieval
* Improve **semantic chunking with section detection**

### ⚡ Performance Enhancements

* Caching embeddings and models
* Parallel processing for PDF ingestion
* Streaming responses

### 🎨 UI/UX Improvements

* ChatGPT-style interface
* Suggested questions
* Better source visualization

### 📊 Advanced Features

* Multi-document comparison
* Export answers (PDF/Markdown)
* Confidence scoring

---

## 🛠️ Installation & Setup

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

Create `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run app:

```bash
streamlit run app.py
```

---

## ☁️ Deployment

Deployed using **Streamlit Cloud**

Steps:

1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Add `GROQ_API_KEY` in Secrets
4. Deploy

---

## 📈 Project Status

✅ Fully functional Multimodal RAG system
🔄 Actively improvable to production-grade

---

## 💡 Summary

This project demonstrates a **real-world implementation of Multimodal RAG**, combining:

* Retrieval systems
* Large Language Models
* Vision understanding
* Vector databases

It showcases how modern AI systems can **analyze and reason over complex documents** beyond plain text.



---
