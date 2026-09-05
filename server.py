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

# Initial threshold based on our retrieval testing.
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
    score_threshold: float | None = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    score_threshold: float | None = None


# ============================================================
# Embedding Helper
# ============================================================


def create_embedding(text: str):
    return model.encode(text).tolist()


# ============================================================
# Memory Storage
# ============================================================


def store_memory(
    text: str,
    metadata: dict | None = None,
):
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


def store_knowledge(
    text: str,
    metadata: dict | None = None,
):
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


def split_large_text(
    text: str,
    chunk_size: int,
):
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
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

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

    pages = []

    with pdfplumber.open(io.BytesIO(content)) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):

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

            return {"error": ("File must be a UTF-8 text file " "or PDF")}

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
def search_memory_endpoint(
    request: SearchRequest,
):

    threshold = (
        DEFAULT_SCORE_THRESHOLD
        if request.score_threshold is None
        else request.score_threshold
    )

    points = search_memory(
        query=request.query,
        limit=request.limit,
        score_threshold=threshold,
    )

    return {
        "collection": MEMORY_COLLECTION,
        "score_threshold": threshold,
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
def search_knowledge_endpoint(
    request: SearchRequest,
):

    threshold = (
        DEFAULT_SCORE_THRESHOLD
        if request.score_threshold is None
        else request.score_threshold
    )

    points = search_knowledge(
        query=request.query,
        limit=request.limit,
        score_threshold=threshold,
    )

    return {
        "collection": KNOWLEDGE_COLLECTION,
        "score_threshold": threshold,
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

    threshold = (
        DEFAULT_SCORE_THRESHOLD
        if request.score_threshold is None
        else request.score_threshold
    )

    # --------------------------------------------------------
    # Search Memory
    # --------------------------------------------------------

    memory_points = search_memory(
        query=request.query,
        limit=request.top_k,
        score_threshold=threshold,
    )

    # --------------------------------------------------------
    # Search Knowledge
    # --------------------------------------------------------

    knowledge_points = search_knowledge(
        query=request.query,
        limit=request.top_k,
        score_threshold=threshold,
    )

    # --------------------------------------------------------
    # Build Memory Context
    # --------------------------------------------------------

    memory_context_parts = []

    for point in memory_points:

        text = point.payload.get(
            "text",
            "",
        )

        score = float(point.score)

        memory_context_parts.append(f"[Score: {score:.3f}]\n{text}")

    memory_context = (
        "\n---\n".join(memory_context_parts)
        if memory_context_parts
        else "No relevant memory found."
    )

    # --------------------------------------------------------
    # Build Knowledge Context
    # --------------------------------------------------------

    knowledge_context_parts = []

    for point in knowledge_points:

        text = point.payload.get(
            "text",
            "",
        )

        score = float(point.score)

        metadata = []

        if point.payload.get("file_name"):

            metadata.append(f"File: {point.payload['file_name']}")

        if point.payload.get("page") is not None:

            metadata.append(f"Page: {point.payload['page']}")

        if point.payload.get("chunk_index") is not None:

            metadata.append(f"Chunk: {point.payload['chunk_index']}")

        metadata_text = ""

        if metadata:

            metadata_text = "\n" + " | ".join(metadata)

        knowledge_context_parts.append(
            f"[Score: {score:.3f}]" f"{metadata_text}\n" f"{text}"
        )

    knowledge_context = (
        "\n---\n".join(knowledge_context_parts)
        if knowledge_context_parts
        else "No relevant knowledge found."
    )

    # --------------------------------------------------------
    # Combined Context
    # --------------------------------------------------------

    context = f"""
--- User Memory ---
{memory_context}

--- Knowledge ---
{knowledge_context}
"""

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use the provided memory and knowledge context
when it is relevant to the user's question.

--- Context ---
{context}
--- End Context ---

Question:
{request.query}

Instructions:
- Use relevant user memory when it helps.
- Use relevant knowledge when it helps.
- Do not assume that every retrieved item is relevant.
- If the context does not contain the answer, use your
  general knowledge when appropriate.
- Do not invent information from the context.
- Keep your answer concise and natural.
"""

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

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
        "score_threshold": threshold,
        "memory_results": len(memory_points),
        "knowledge_results": len(knowledge_points),
        "context_used": context,
    }
