from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import ollama
import pdfplumber
import io
import re

app = FastAPI()


# ============================================================
# Configuration
# ============================================================

MEMORY_COLLECTION = "memory"
KNOWLEDGE_COLLECTION = "knowledge"

VECTOR_SIZE = 384

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Initial threshold based on our score evaluation.
# This is NOT considered the final universal threshold.
DEFAULT_SCORE_THRESHOLD = 0.30


# ============================================================
# Qdrant
# ============================================================

qdrant = QdrantClient(":memory:")


qdrant.recreate_collection(
    collection_name=MEMORY_COLLECTION,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE,
    ),
)


qdrant.recreate_collection(
    collection_name=KNOWLEDGE_COLLECTION,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE,
    ),
)


# ============================================================
# Embedding Model
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# Request Models
# ============================================================


class MemoryRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 3

    # Optional so we can experiment with different thresholds.
    # If omitted, DEFAULT_SCORE_THRESHOLD is used.
    score_threshold: float | None = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


# ============================================================
# Embedding Helper
# ============================================================


def create_embedding(text: str):
    """
    Convert text into a vector using SentenceTransformer.
    """
    return model.encode(text).tolist()


# ============================================================
# Memory Storage
# ============================================================


def store_memory(text: str, metadata: dict | None = None):
    """
    Store a single memory in the memory collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    payload = {
        "text": text,
        "type": "memory",
    }

    if metadata:
        payload.update(metadata)

    qdrant.upsert(
        collection_name=MEMORY_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )

    return point_id


# ============================================================
# Knowledge Storage
# ============================================================


def store_knowledge(text: str, metadata: dict | None = None):
    """
    Store a single knowledge chunk in the knowledge collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    payload = {
        "text": text,
        "type": "knowledge",
    }

    if metadata:
        payload.update(metadata)

    qdrant.upsert(
        collection_name=KNOWLEDGE_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )

    return point_id


# ============================================================
# Memory Search
# ============================================================


def search_memory(
    query: str,
    limit: int = 3,
    score_threshold: float | None = None,
):
    """
    Search only the memory collection.

    Qdrant filters results using score_threshold.
    """

    query_embedding = create_embedding(query)

    threshold = DEFAULT_SCORE_THRESHOLD if score_threshold is None else score_threshold

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=threshold,
    )

    return results.points


# ============================================================
# Knowledge Search
# ============================================================


def search_knowledge(
    query: str,
    limit: int = 3,
    score_threshold: float | None = None,
):
    """
    Search only the knowledge collection.

    Qdrant filters results using score_threshold.
    """

    query_embedding = create_embedding(query)

    threshold = DEFAULT_SCORE_THRESHOLD if score_threshold is None else score_threshold

    results = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=threshold,
    )

    return results.points


# ============================================================
# Text Chunking
# ============================================================


def split_large_text(text: str, chunk_size: int):
    """
    Split a very large piece of text into smaller pieces.
    """

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Paragraph-aware chunking.

    Tries to keep paragraphs together while respecting
    the approximate chunk size.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # If a paragraph itself is larger than the target size,
        # split it separately.
        if len(paragraph) > chunk_size:

            if current:
                chunks.append(current.strip())
                current = ""

            large_chunks = split_large_text(
                paragraph,
                chunk_size,
            )

            chunks.extend(large_chunks)

            continue

        candidate = paragraph if not current else current + "\n\n" + paragraph

        if len(candidate) <= chunk_size:

            current = candidate

        else:

            if current:
                chunks.append(current.strip())

            # Add overlap from the previous chunk.
            overlap_text = ""

            if overlap > 0 and current:
                overlap_text = current[-overlap:]

            current = overlap_text + "\n\n" + paragraph if overlap_text else paragraph

    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# PDF Extraction
# ============================================================


def extract_pdf_pages(content: bytes):
    """
    Extract PDF text page by page.

    Returns:

    [
        {
            "page": 1,
            "text": "..."
        },
        ...
    ]
    """

    pages = []

    with pdfplumber.open(io.BytesIO(content)) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            page_text = page.extract_text()

            if page_text and page_text.strip():

                pages.append(
                    {
                        "page": page_number,
                        "text": page_text.strip(),
                    }
                )

    return pages


# ============================================================
# Text File Chunk Creation
# ============================================================


def create_text_chunks(text: str):
    """
    Create knowledge chunks for a text file.
    """

    chunks = chunk_text(text)

    results = []

    for chunk_index, chunk in enumerate(chunks):

        results.append(
            {
                "text": chunk,
                "chunk_index": chunk_index,
            }
        )

    return results


# ============================================================
# PDF Chunk Creation
# ============================================================


def create_pdf_chunks(content: bytes):
    """
    Extract PDF page by page and chunk each page independently.

    Chunks do not intentionally cross page boundaries.
    """

    pages = extract_pdf_pages(content)

    results = []

    global_chunk_index = 0

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = chunk_text(page_text)

        for page_chunk_index, chunk in enumerate(page_chunks):

            results.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "chunk_index": global_chunk_index,
                    "page_chunk_index": page_chunk_index,
                }
            )

            global_chunk_index += 1

    return results


# ============================================================
# /store
# ============================================================


@app.post("/store")
def add_memory(request: MemoryRequest):

    if not request.text.strip():
        return {"error": "Memory text cannot be empty"}

    point_id = store_memory(
        text=request.text,
    )

    return {
        "status": "stored",
        "collection": MEMORY_COLLECTION,
        "id": point_id,
    }


# ============================================================
# /add_memory_file_upload
# ============================================================


@app.post("/add_memory_file_upload")
async def add_memory_file_upload(
    file: UploadFile = File(...),
):

    if not file.filename:
        return {"error": "Filename is required"}

    filename = file.filename.lower()

    content = await file.read()

    stored_ids = []

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if filename.endswith(".pdf"):

        chunks = create_pdf_chunks(content)

        if not chunks:
            return {"error": "PDF contains no extractable text"}

        for chunk_data in chunks:

            point_id = store_knowledge(
                text=chunk_data["text"],
                metadata={
                    "file_name": file.filename,
                    "page": chunk_data["page"],
                    "chunk_index": chunk_data["chunk_index"],
                    "page_chunk_index": chunk_data["page_chunk_index"],
                },
            )

            stored_ids.append(point_id)

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    else:

        try:
            text = content.decode("utf-8")

        except UnicodeDecodeError:

            return {"error": "File must be a UTF-8 text file or PDF"}

        if not text.strip():
            return {"error": "File is empty"}

        chunks = create_text_chunks(text)

        for chunk_data in chunks:

            point_id = store_knowledge(
                text=chunk_data["text"],
                metadata={
                    "file_name": file.filename,
                    "chunk_index": chunk_data["chunk_index"],
                },
            )

            stored_ids.append(point_id)

    return {
        "status": "stored",
        "collection": KNOWLEDGE_COLLECTION,
        "file_name": file.filename,
        "chunks_stored": len(stored_ids),
        "ids": stored_ids,
    }


# ============================================================
# /search_memory
# ============================================================


@app.post("/search_memory")
def search_memory_endpoint(request: SearchRequest):

    points = search_memory(
        query=request.query,
        limit=request.limit,
        score_threshold=request.score_threshold,
    )

    return {
        "collection": MEMORY_COLLECTION,
        "score_threshold": (
            DEFAULT_SCORE_THRESHOLD
            if request.score_threshold is None
            else request.score_threshold
        ),
        "results": [
            {
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "score": float(point.score),
            }
            for point in points
        ],
    }


# ============================================================
# /search_knowledge
# ============================================================


@app.post("/search_knowledge")
def search_knowledge_endpoint(request: SearchRequest):

    points = search_knowledge(
        query=request.query,
        limit=request.limit,
        score_threshold=request.score_threshold,
    )

    return {
        "collection": KNOWLEDGE_COLLECTION,
        "score_threshold": (
            DEFAULT_SCORE_THRESHOLD
            if request.score_threshold is None
            else request.score_threshold
        ),
        "results": [
            {
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "file_name": point.payload.get("file_name"),
                "page": point.payload.get("page"),
                "chunk_index": point.payload.get("chunk_index"),
                "page_chunk_index": point.payload.get("page_chunk_index"),
                "score": float(point.score),
            }
            for point in points
        ],
    }


# ============================================================
# /chat
# ============================================================


@app.post("/chat")
def chat(request: ChatRequest):

    # --------------------------------------------------------
    # NOTE:
    # /chat is intentionally still using the existing
    # memory-only retrieval behavior.
    #
    # We will update /chat after threshold retrieval has
    # been independently tested.
    # --------------------------------------------------------

    query_embedding = create_embedding(request.query)

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        limit=request.top_k,
    )

    context_parts = []

    for point in results.points:

        part = point.payload.get("text", "")

        if point.payload.get("file_name"):
            part += f"\nFile: " f"{point.payload['file_name']}"

        context_parts.append(part)

    context = (
        "\n---\n".join(context_parts) if context_parts else "No relevant memory found."
    )

    prompt = f"""
You are a helpful AI assistant.

Use the following relevant information to answer
the user's question.

--- Memory context ---
{context}
--- End of memory context ---

Question: {request.query}

Instructions:
- Use memory if it helps.
- If memory is not relevant, answer based on general knowledge.
- Keep your answer concise and natural.
"""

    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return {
        "response": response["message"]["content"],
        "context_used": context,
    }
