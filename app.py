import streamlit as st
import os
import io
import json
import time
import base64
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)

# Use updated import path (langchain_huggingface preferred, fallback to community)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ==================== EMBEDDINGS ====================
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

embedding_model = load_embeddings()

# ==================== CONFIG ====================
load_dotenv()

# Groq API (free Llama inference)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# Model selection — Groq offers these Llama models for free
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")

IMAGES_DIR = Path("extracted_images")
CHROMA_DIR = "chroma_db"
HISTORY_FILE = "chat_history.json"
MIN_IMAGE_SIZE = 50  # Skip images smaller than 50x50 px

IMAGES_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Multimodal RAG Chat (Llama)",
    page_icon="🧠",
    layout="wide",
)

# ==================== PREMIUM CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(160deg, #0a0e1a 0%, #0d1321 40%, #121a2e 100%);
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c7d2fe;
}

.main-header {
    text-align: center;
    padding: 2rem 0 0.5rem;
    position: relative;
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa, #c084fc);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite;
    margin-bottom: 0.25rem;
}

.main-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 300;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.glow-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #a78bfa, #6366f1, transparent);
    margin: 0.5rem auto 1.5rem;
    width: 60%;
    border-radius: 2px;
    animation: glowPulse 3s ease-in-out infinite;
}

@keyframes glowPulse {
    0%, 100% { opacity: 0.5; filter: blur(0px); }
    50% { opacity: 1; filter: blur(1px); }
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 12px;
    animation: fadeInUp 0.4s ease-out;
    backdrop-filter: blur(12px);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"][aria-label="user"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.15));
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-left: 3px solid #6366f1;
}

[data-testid="stChatMessage"][aria-label="assistant"] {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-left: 3px solid #22d3ee;
}

.stButton > button {
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
    background: linear-gradient(135deg, #818cf8, #a78bfa);
}

.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    margin-bottom: 8px;
}

.stat-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    transform: translateY(-2px);
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
}

.stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 8px;
    padding: 6px 12px;
    margin: 4px;
    font-size: 0.8rem;
    color: #c7d2fe;
    transition: all 0.2s ease;
}

.source-badge:hover {
    background: rgba(99, 102, 241, 0.25);
    border-color: rgba(99, 102, 241, 0.4);
}

[data-testid="stFileUploader"] {
    border-radius: 12px;
}

[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.4);
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: rgba(99, 102, 241, 0.6);
    background: rgba(99, 102, 241, 0.05);
}

[data-testid="stExpander"] {
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.3);
}

[data-testid="stChatInput"] textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
}

.stSpinner > div {
    border-top-color: #6366f1 !important;
}

.stAlert {
    border-radius: 12px;
}

.welcome-card {
    background: rgba(30, 41, 59, 0.35);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin: 2rem auto;
    max-width: 600px;
}

.welcome-card h3 {
    color: #c7d2fe;
    margin-bottom: 0.5rem;
}

.welcome-card p {
    color: #94a3b8;
    font-size: 0.9rem;
}

.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 1.5rem;
}

.feature-item {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 10px;
    padding: 12px;
    font-size: 0.82rem;
    color: #cbd5e1;
    transition: all 0.2s ease;
}

.feature-item:hover {
    border-color: rgba(99, 102, 241, 0.4);
    background: rgba(99, 102, 241, 0.12);
    transform: translateY(-1px);
}

.feature-icon {
    font-size: 1.3rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ==================== GROQ / LLAMA API HELPERS ====================
def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def check_groq_available() -> bool:
    """Check if Groq API key is configured."""
    return bool(GROQ_API_KEY)


def llama_generate_groq(prompt: str, images: list = None, model: str = None) -> str:
    """
    Generate response using Groq API (free Llama inference).
    images: list of PIL Image objects (optional, for vision models)
    """
    if not GROQ_API_KEY:
        raise Exception(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        )

    if model is None:
        model = GROQ_VISION_MODEL if images else GROQ_TEXT_MODEL

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build messages
    if images:
        # Multimodal message format for vision models
        content = []

        # Add images first (Groq vision supports base64 images)
        for img in images[:3]:
            b64 = image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            GROQ_BASE_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_body = response.json()
            error_detail = error_body.get("error", {}).get("message", "")
        except Exception:
            pass

        status = response.status_code
        if status == 429:
            raise Exception(f"Groq rate limit reached. Wait a moment and try again. {error_detail}")
        elif status == 401:
            raise Exception("Invalid GROQ_API_KEY. Check your key at https://console.groq.com")
        elif status == 400 and images:
            # Vision model may not be available — fall back to text-only
            raise Exception(f"Groq vision request failed: {error_detail}")
        else:
            raise Exception(f"Groq API error (HTTP {status}): {error_detail or str(e)}")
    except requests.exceptions.Timeout:
        raise Exception("Groq request timed out. Try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to Groq API. Check your internet connection.")
    except Exception as e:
        if "Groq" in str(e):
            raise
        raise Exception(f"Groq API error: {str(e)}")


def llama_generate(prompt: str, images: list = None, max_retries: int = 3) -> str:
    """
    Unified function to generate response using Groq Llama backend.
    Includes retry logic for transient errors (rate limits, timeouts).
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return llama_generate_groq(prompt, images)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "quota" in err_str:
                wait = 10 * (2 ** attempt)
                st.toast(f"Rate limit hit — waiting {wait}s before retry {attempt+1}/{max_retries}…")
                time.sleep(wait)
            elif "vision" in err_str and images:
                # If vision failed, retry without images (text-only fallback)
                st.toast("Vision model unavailable — falling back to text-only…")
                try:
                    return llama_generate_groq(prompt, images=None)
                except Exception as fallback_e:
                    last_error = fallback_e
                    break
            elif attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise

    raise last_error


# ==================== SESSION STATE ====================
def init_session_state():
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "images": {},
        "processing_stats": None,
        "pdf_processed": False,
        "history_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ==================== PERSISTENCE ====================
def save_chat_history():
    """Save chat history to JSON (excluding PIL objects)."""
    serializable = []
    for chat in st.session_state.chat_history:
        serializable.append({
            "question": chat["question"],
            "answer": chat["answer"],
            "sources": chat.get("sources", []),
            "image_paths": chat.get("image_paths", []),
            "timestamp": chat.get("timestamp", ""),
        })
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save chat history: {e}")


def load_chat_history():
    """Load chat history from JSON file — only once per session."""
    if st.session_state.history_loaded:
        return
    st.session_state.history_loaded = True

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    st.session_state.chat_history = data
        except (json.JSONDecodeError, Exception):
            st.session_state.chat_history = []


load_chat_history()


# ==================== PDF PROCESSING ====================
def extract_tables_from_page(page):
    """Extract tables from a PDF page as Markdown strings."""
    tables_md = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            try:
                df = table.to_pandas()
                if not df.empty:
                    tables_md.append(df.to_markdown(index=False))
            except Exception:
                rows = table.extract()
                if rows:
                    row_strs = [" | ".join(str(c) for c in row) for row in rows]
                    tables_md.append("\n".join(row_strs))
    except AttributeError:
        pass
    except Exception:
        pass
    return tables_md


def caption_image(image: Image.Image) -> str:
    """
    Caption an image using Llama vision model via Groq.
    """
    prompt = (
        "Describe this image in detail. Include: what it shows, any visible text, "
        "data/chart/diagram explanations, colours, and layout. Be thorough but concise."
    )

    try:
        return llama_generate(prompt, images=[image])
    except Exception as e:
        return f"[Image — captioning failed: {str(e)[:120]}]"


def get_pdf_content(pdf_docs, progress_callback=None):
    """
    Full multimodal extraction pipeline.
    Returns (documents, images_dict, stats).
    """
    documents = []
    images = {}
    stats = {"pages": 0, "images": 0, "tables": 0, "pdfs": len(pdf_docs)}

    # Clear old images safely
    for old in IMAGES_DIR.glob("*"):
        try:
            old.unlink()
        except Exception:
            pass

    # Pre-count total pages
    total_pages = 0
    pdf_bytes_list = []
    for pdf in pdf_docs:
        raw = pdf.read()
        pdf_bytes_list.append((pdf.name, raw))
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
            total_pages += len(doc)
            doc.close()
        except Exception:
            pass

    processed_pages = 0

    for pdf_name, raw_bytes in pdf_bytes_list:
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
        except Exception as e:
            st.warning(f"Could not open {pdf_name}: {e}")
            continue

        pdf_stem = Path(pdf_name).stem

        for page_num, page in enumerate(doc):
            # --- Text ---
            text = page.get_text()
            if text.strip():
                documents.append({
                    "text": text,
                    "source": pdf_name,
                    "page": page_num + 1,
                    "type": "text",
                })

            # --- Tables ---
            tables = extract_tables_from_page(page)
            for tbl_md in tables:
                documents.append({
                    "text": f"[TABLE from {pdf_name} Page {page_num + 1}]\n{tbl_md}",
                    "source": pdf_name,
                    "page": page_num + 1,
                    "type": "table",
                })
                stats["tables"] += 1

            # --- Images ---
            img_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(img_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    if not image_bytes or len(image_bytes) < 100:
                        continue

                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    w, h = image.size
                    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                        continue

                    img_filename = f"{pdf_stem}_p{page_num + 1}_img{img_idx}.png"
                    img_path = str(IMAGES_DIR / img_filename)
                    image.save(img_path)
                    images[img_path] = image

                    # Caption with Llama Vision via Groq
                    caption = caption_image(image)

                    documents.append({
                        "text": f"[IMAGE from {pdf_name} Page {page_num + 1}]\n{caption}",
                        "source": pdf_name,
                        "page": page_num + 1,
                        "type": "image",
                        "image_path": img_path,
                    })
                    stats["images"] += 1

                except Exception:
                    continue

            processed_pages += 1
            stats["pages"] = processed_pages
            if progress_callback:
                pct = processed_pages / max(total_pages, 1)
                progress_callback(pct, f"Processing page {processed_pages}/{total_pages}…")

        doc.close()

    return documents, images, stats


def get_text_chunks(documents):
    texts = []
    metadatas = []

    for doc in documents:
        raw = doc["text"]
        if not raw or not raw.strip():
            continue

        #  HYBRID LOGIC
        if doc["type"] == "text":
            try:
                chunks = semantic_splitter.split_text(raw)

                # safety check
                if not chunks or len(chunks) < 2:
                    chunks = recursive_splitter.split_text(raw)

            except Exception:
                chunks = recursive_splitter.split_text(raw)

        else:
            # tables & images untouched
            chunks = [raw]

        for chunk in chunks:
            if not chunk.strip():
                continue

            texts.append(chunk)

            meta = {
                "source": doc["source"],
                "page": str(doc["page"]),
                "type": doc["type"],
            }

            if "image_path" in doc:
                meta["image_path"] = doc["image_path"]

            metadatas.append(meta)

    return texts, metadatas


def get_vector_store(texts, metadatas):
    """Build ChromaDB vector store."""
    if not texts:
        raise ValueError("No text content to embed.")

    embeddings = embedding_model

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="multimodal_rag",
    )

    return vector_store


def reload_vector_store_from_disk():
    """Reload existing vector store from disk."""
    try:
        if not os.path.exists(CHROMA_DIR):
            return None

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vs = Chroma(
            collection_name="multimodal_rag",
            embedding_function=embeddings,
        )

        if vs._collection.count() > 0:
            return vs
    except Exception:
        pass
    return None

    if vs:
        st.session_state.vector_store = vs
        st.session_state.pdf_processed = True

    # Also reload images dict from disk
    if not st.session_state.images:
        for img_path in IMAGES_DIR.glob("*.png"):
            try:
                st.session_state.images[str(img_path)] = Image.open(img_path)
            except Exception:
                pass


# ==================== QUERY ====================
def build_conversation_context(history, max_turns=5):
    """Return a string of the last N Q/A turns for multi-turn context."""
    recent = history[-max_turns:]
    parts = []
    for chat in recent:
        parts.append(f"User: {chat['question']}")
        ans = chat["answer"]
        if len(ans) > 500:
            ans = ans[:500] + "…"
        parts.append(f"Assistant: {ans}")
    return "\n".join(parts)


def multimodal_query(question, vector_store, chat_history):
    """
    Retrieve relevant chunks, build a multimodal prompt,
    and generate an answer with Llama via Groq.
    Returns (answer_text, sources_list, image_paths_list).
    """
    # Use MMR search for better diversity
    try:
        docs = vector_store.max_marginal_relevance_search(question, k=6, fetch_k=20)
    except Exception:
        docs = vector_store.similarity_search(question, k=6)

    if not docs:
        return "⚠️ No relevant content found in the uploaded documents.", [], []

    context_parts = []
    sources = []
    image_paths = []
    seen = set()

    for doc in docs:
        context_parts.append(doc.page_content)

        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("type", "text")
        key = f"{src}|{page}|{dtype}"

        if key not in seen:
            icon = {"text": "📄", "table": "📊", "image": "🖼️"}.get(dtype, "📄")
            sources.append(f"{icon} {src} — Page {page}")
            seen.add(key)

        img_path = doc.metadata.get("image_path", "")
        if img_path and os.path.exists(img_path) and img_path not in image_paths:
            image_paths.append(img_path)

    context = "\n\n---\n\n".join(context_parts)
    conv_ctx = build_conversation_context(chat_history)

    prompt = (
        "You are an expert AI assistant analysing multimodal documents (text, tables, images). "
        "Answer the question thoroughly using ONLY the provided context below.\n"
        "• If the context contains image descriptions, reference them naturally in your answer.\n"
        "• If tables are present, use them for precise data extraction.\n"
        "• If the answer is not in the context, clearly say so — do not hallucinate.\n"
        "• Use Markdown formatting: headers, bullets, bold for key terms.\n\n"
    )
    if conv_ctx:
        prompt += f"**Previous conversation:**\n{conv_ctx}\n\n"
    prompt += f"**Document Context:**\n{context}\n\n**Question:** {question}"

    # Load images for vision model (limit to 3)
    images_to_send = []
    for p in image_paths[:3]:
        try:
            img = Image.open(p).convert("RGB")
            images_to_send.append(img)
        except Exception:
            continue

    try:
        response = llama_generate(prompt, images=images_to_send if images_to_send else None)
        return response, sources, image_paths
    except Exception as e:
        return f" Error generating response: {e}", sources, image_paths


# ==================== UI — HEADER ====================
st.markdown("""
<div class="main-header">
    <h1>Multimodal RAG Chat</h1>
    <p>Upload PDFs and ask questions about text, tables, charts &amp; images</p>
    <div class="glow-line"></div>
</div>
""", unsafe_allow_html=True)


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("##  Document Upload")

    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"],
        help="Upload one or more PDFs to analyse",
    )

    if st.button(" Process Documents", use_container_width=True):
        if pdf_docs:
            # Check backend availability first
            if not check_groq_available():
                st.error("Cannot process: GROQ_API_KEY not set. Get a free key at https://console.groq.com")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def _progress(pct, msg):
                    progress_bar.progress(min(float(pct), 1.0))
                    status_text.text(msg)

                try:
                    status_text.text("📄 Extracting content & captioning images…")
                    documents, images, stats = get_pdf_content(pdf_docs, _progress)

                    if not documents:
                        st.error("No content could be extracted from the uploaded PDFs.")
                    else:
                        status_text.text("✂️ Chunking documents…")
                        progress_bar.progress(0.80)
                        texts, metadatas = get_text_chunks(documents)
                        stats["chunks"] = len(texts)

                        status_text.text("🧮 Building vector embeddings…")
                        progress_bar.progress(0.90)
                        st.session_state.vector_store = get_vector_store(texts, metadatas)

                        st.session_state.images = images
                        st.session_state.processing_stats = stats
                        st.session_state.pdf_processed = True

                        progress_bar.progress(1.0)
                        status_text.text("✅ Complete!")
                        st.success(
                            f"✅ Processed {stats['pdfs']} PDF(s)  |  "
                            f"{stats['pages']} pages  |  "
                            f"{stats['images']} images  |  "
                            f"{stats['tables']} tables"
                        )

                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    progress_bar.empty()
                    status_text.empty()
        else:
            st.warning("Please upload at least one PDF.")

    # ---- Stats Dashboard ----
    if st.session_state.processing_stats:
        stats = st.session_state.processing_stats
        st.markdown("---")
        st.markdown("### 📊 Processing Stats")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats.get("pages", 0)}</div>'
                f'<div class="stat-label">Pages</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats.get("images", 0)}</div>'
                f'<div class="stat-label">Images</div></div>',
                unsafe_allow_html=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats.get("tables", 0)}</div>'
                f'<div class="stat-label">Tables</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats.get("chunks", 0)}</div>'
                f'<div class="stat-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )

    # ---- Image Gallery ----
    if st.session_state.images:
        st.markdown("---")
        st.markdown("### 🖼️ Extracted Images")
        img_paths = list(st.session_state.images.keys())
        cols = st.columns(2)
        for i, p in enumerate(img_paths[:6]):
            with cols[i % 2]:
                if os.path.exists(p):
                    st.image(p, use_container_width=True, caption=Path(p).stem)
        if len(img_paths) > 6:
            st.caption(f"+{len(img_paths) - 6} more images")

    # ---- Summarize ----
    st.markdown("---")
    if st.button("📄 Summarize Documents", use_container_width=True):
        if st.session_state.vector_store:
            with st.spinner("Generating summary…"):
                try:
                    docs = st.session_state.vector_store.similarity_search(
                        "summary overview main topics key points conclusions", k=8
                    )
                    context = "\n\n".join([d.page_content for d in docs])

                    summary_prompt = (
                        "Provide a comprehensive, well-structured summary of these documents. "
                        "Use Markdown with headers, bullet points, and sections. "
                        "Include key findings, data points, and visual descriptions.\n\n"
                        + context
                    )

                    # Gather images for summary
                    summary_images = []
                    for d in docs:
                        ip = d.metadata.get("image_path", "")
                        if ip and os.path.exists(ip) and len(summary_images) < 3:
                            try:
                                img = Image.open(ip).convert("RGB")
                                summary_images.append(img)
                            except Exception:
                                pass

                    resp = llama_generate(
                        summary_prompt,
                        images=summary_images if summary_images else None
                    )
                    st.markdown("### 📋 Document Summary")
                    st.markdown(resp)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
        else:
            st.warning("Process PDFs first!")

    # ---- Actions ----
    st.markdown("---")
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
    with ac2:
        if st.button("🔄 Reset All", use_container_width=True):
            import shutil
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            if os.path.exists(CHROMA_DIR):
                try:
                    shutil.rmtree(CHROMA_DIR)
                except Exception:
                    pass
            for img in IMAGES_DIR.glob("*"):
                try:
                    img.unlink()
                except Exception:
                    pass
            st.rerun()


# ==================== CHAT DISPLAY ====================
if not st.session_state.chat_history and not st.session_state.pdf_processed:
    st.markdown("""
    <div class="welcome-card">
        <h3>Welcome to Multimodal RAG Chat</h3>
        <p>Upload PDF documents and ask anything — including questions about
        images, charts, and tables inside them.</p>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">📄</div>
                Text Understanding
            </div>
            <div class="feature-item">
                <div class="feature-icon">🖼️</div>
                Image Analysis
            </div>
            <div class="feature-item">
                <div class="feature-icon">📊</div>
                Table Extraction
            </div>
            <div class="feature-item">
                <div class="feature-icon">💬</div>
                Multi-turn Chat
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.pdf_processed and not st.session_state.chat_history:
    pass
    

for chat in st.session_state.chat_history:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(chat["question"])

    with st.chat_message("assistant", avatar="🦙"):
        st.markdown(chat["answer"])

        img_paths = chat.get("image_paths", [])
        if img_paths:
            valid_imgs = [p for p in img_paths if os.path.exists(p)]
            if valid_imgs:
                st.markdown("**Referenced Images:**")
                icols = st.columns(min(len(valid_imgs), 3))
                for i, p in enumerate(valid_imgs[:3]):
                    with icols[i]:
                        st.image(p, use_container_width=True)

        sources = chat.get("sources", [])
        if sources:
            with st.expander("📚 View Sources", expanded=False):
                html = "".join(f'<span class="source-badge">{s}</span>' for s in sources)
                st.markdown(html, unsafe_allow_html=True)

        ts = chat.get("timestamp", "")
        if ts:
            st.caption(f"🕐 {ts}")


# ==================== CHAT INPUT ====================
user_question = st.chat_input("Ask anything about your documents…")

if user_question:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process PDF documents first.")
    else:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking…"):
                response, sources, image_paths = multimodal_query(
                    user_question,
                    st.session_state.vector_store,
                    st.session_state.chat_history,
                )

            st.markdown(response)

            if image_paths:
                valid_imgs = [p for p in image_paths if os.path.exists(p)]
                if valid_imgs:
                    st.markdown("**Referenced Images:**")
                    icols = st.columns(min(len(valid_imgs), 3))
                    for i, p in enumerate(valid_imgs[:3]):
                        with icols[i]:
                            st.image(p, use_container_width=True)

            if sources:
                with st.expander("📚 View Sources", expanded=False):
                    html = "".join(f'<span class="source-badge">{s}</span>' for s in sources)
                    st.markdown(html, unsafe_allow_html=True)

            timestamp = datetime.now().strftime("%I:%M %p")
            st.caption(f"🕐 {timestamp}")

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response,
            "sources": sources,
            "image_paths": image_paths,
            "timestamp": datetime.now().strftime("%I:%M %p, %b %d"),
        })

        save_chat_history()
