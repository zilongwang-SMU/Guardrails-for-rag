import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, List, Dict, cast

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from transformers import pipeline


# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

PDF_PATH = DATA_DIR / "DH-Chapter2.pdf"
OUTPUT_FILE = OUTPUT_DIR / "results.txt"

REQUIRED_QUERIES = [
    "What is Crosswalk guards?",
    "What to do if moving through an intersection with a green signal?",
    "What to do when approached by an emergency vehicle?",
]

# CPU-friendly model (no API key)
HF_MODEL = "google/flan-t5-base"

# Retrieval settings
TOP_K = 3


# ============================================================
# Direct Jina Embeddings (robust batching + retry)
# ============================================================
class JinaDirectEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v2-base-en",
        batch_size: int = 8,
        max_retries: int = 5,
        retry_sleep_sec: float = 1.0,
        max_chars: int = 4000,
    ):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.url = "https://api.jina.ai/v1/embeddings"
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_sleep_sec = retry_sleep_sec
        self.max_chars = max_chars

        if not self.api_key:
            raise ValueError("Missing JINA_API_KEY (empty). Check your .env file.")

    def _post_with_retry(self, batch: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": batch}

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            resp = requests.post(self.url, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()["data"]
                data_sorted = sorted(data, key=lambda x: x["index"])
                return [item["embedding"] for item in data_sorted]

            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"{resp.status_code}: {resp.text}"
                time.sleep(self.retry_sleep_sec * attempt)
                continue

            raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text}")

        raise RuntimeError(f"Jina API failed after retries. Last error: {last_err}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        clean = []
        for t in texts:
            t = "" if t is None else str(t)
            t = t.replace("\x00", "")
            if len(t) > self.max_chars:
                t = t[: self.max_chars]
            clean.append(t)

        out: list[list[float]] = []
        for i in range(0, len(clean), self.batch_size):
            batch = clean[i : i + self.batch_size]
            out.extend(self._post_with_retry(batch))
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


# ============================================================
# Helpers
# ============================================================
def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)


def load_env_keys():
    load_dotenv(PROJECT_ROOT / ".env")
    jina_key = os.getenv("JINA_API_KEY")
    if not jina_key:
        raise ValueError("Missing JINA_API_KEY in .env")
    return jina_key.strip()


def load_pdfs_from_data_dir():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}. Put DH-Chapter2.pdf there.")
    return pdf_files


def build_or_load_vectorstore(jina_api_key: str):
    embedding = JinaDirectEmbeddings(api_key=jina_api_key)

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("[INFO] Loading existing ChromaDB from chroma_db/ ...")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding,
        )

    print("[INFO] Building new ChromaDB from PDFs in data/ ...")
    pdf_files = load_pdfs_from_data_dir()

    all_docs = []
    for pdf in pdf_files:
        print(f"[INFO] Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    print(f"[INFO] Total chunks: {len(chunks)}")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(CHROMA_DIR),
    )

    print("[INFO] ChromaDB saved to chroma_db/")
    return vs


def make_answer_fn(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # HuggingFace pipeline (flan-t5 uses text2text-generation)
    hf = pipeline("text2text-generation", model=HF_MODEL)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question using ONLY the context. "
         "If the answer is not in the context, reply exactly: Not found in the document.\n\n"
         "Context:\n{context}"),
        ("human", "Question: {question}\nAnswer:")
    ])

    def answer_question(question: str):
        docs = retriever.invoke(question)
        if not docs:
            return "Not found in the document.", []

        context_text = "\n\n".join(d.page_content for d in docs)

        messages = prompt.format_messages(context=context_text, question=question)
        text_prompt = "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

        out = hf(
            text_prompt,
            max_new_tokens=180,
            do_sample=False,
        )

        # Pylance-safe: cast pipeline output to List[Dict[str, Any]]
        out_list = cast(List[Dict[str, Any]], out)
        answer_text = str(out_list[0].get("generated_text", "")).strip()

        if not answer_text:
            answer_text = "Not found in the document."

        return answer_text, docs

    return answer_question


def write_required_results(answer_fn):
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in REQUIRED_QUERIES:
            ans, src_docs = answer_fn(q)

            f.write(f"QUESTION: {q}\n")
            f.write(f"ANSWER: {ans}\n")

            if src_docs:
                f.write("SOURCES:\n")
                for d in src_docs:
                    page = d.metadata.get("page", "N/A")
                    source = d.metadata.get("source", "N/A")
                    f.write(f" - {Path(source).name} | Page {page}\n")
            else:
                f.write("SOURCES: Not found\n")

            f.write("\n" + "=" * 50 + "\n\n")

    print(f"[OK] Saved required answers to: {OUTPUT_FILE}")


def cli_loop(answer_fn):
    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        ans, src_docs = answer_fn(q)
        print("\nAnswer:\n" + ans)

        if src_docs:
            print("\nSources:")
            for d in src_docs:
                page = d.metadata.get("page", "N/A")
                source = d.metadata.get("source", "N/A")
                print(f"- {Path(source).name} | Page {page}")


def main():
    ensure_dirs()
    jina_api_key = load_env_keys()

    if not PDF_PATH.exists():
        print(f"[WARN] {PDF_PATH} not found.")
        print("[TIP] Put DH-Chapter2.pdf into data/ folder (data/DH-Chapter2.pdf).")
        if not list(DATA_DIR.glob("*.pdf")):
            raise FileNotFoundError("No PDFs in data/. Cannot continue.")

    vectorstore = build_or_load_vectorstore(jina_api_key)
    answer_fn = make_answer_fn(vectorstore)

    write_required_results(answer_fn)
    cli_loop(answer_fn)


if __name__ == "__main__":
    main()
