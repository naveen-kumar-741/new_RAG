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


# ============================================================
# Qdrant
# ============================================================

# In-memory Qdrant for now.
# Persistent Qdrant will be handled later.
qdrant = QdrantClient(":memory:")


# Memory collection
qdrant.recreate_collection(
    collection_name=MEMORY_COLLECTION,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE,
    ),
)


# Knowledge collection
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


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


# ============================================================
# Embedding Helper
# ============================================================

def create_embedding(text: str):
    """
    Convert text into a vector embedding.
    """

    return model.encode(text).tolist()


# ============================================================
# Memory Storage
# ============================================================

def store_memory(
    text: str,
    metadata: dict | None = None,
):
    """
    Store user memory in the memory collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    payload = {
        "text": text,
        "type": "memory",
        **(metadata or {}),
    }

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
    """
    Store document knowledge in the knowledge collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    payload = {
        "text": text,
        "type": "knowledge",
        **(metadata or {}),
    }

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
):
    """
    Search only the memory collection.
    """

    query_embedding = create_embedding(query)

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        limit=limit,
    )

    return results.points


# ============================================================
# Knowledge Search
# ============================================================

def search_knowledge(
    query: str,
    limit: int = 3,
):
    """
    Search only the knowledge collection.
    """

    query_embedding = create_embedding(query)

    results = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=query_embedding,
        limit=limit,
    )

    return results.points


# ============================================================
# Text Chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into paragraph-aware chunks.

    Strategy:
    1. Preserve paragraph boundaries where possible.
    2. Keep chunks around chunk_size characters.
    3. Use overlap when a paragraph is larger than chunk_size.
    4. Fall back to hard splitting for very large paragraphs.
    """

    text = text.strip()

    if not text:
        return []

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Split into paragraphs
    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # Normalize whitespace inside paragraph
        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph,
        ).strip()

        if not paragraph:
            continue

        # ----------------------------------------------------
        # Normal paragraph
        # ----------------------------------------------------

        if len(paragraph) <= chunk_size:

            candidate = (
                f"{current}\n\n{paragraph}"
                if current
                else paragraph
            )

            if len(candidate) <= chunk_size:

                current = candidate

            else:

                if current:
                    chunks.append(current)

                current = paragraph

            continue

        # ----------------------------------------------------
        # Large paragraph
        # ----------------------------------------------------

        if current:

            chunks.append(current)
            current = ""

        start = 0

        while start < len(paragraph):

            end = start + chunk_size

            chunk = paragraph[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Prevent invalid overlap configuration
            step = max(
                1,
                chunk_size - overlap,
            )

            start += step

    # --------------------------------------------------------
    # Store final chunk
    # --------------------------------------------------------

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# PDF Extraction
# ============================================================

def extract_pdf_pages(
    content: bytes,
) -> list[dict]:
    """
    Extract text from each PDF page separately.

    Keeping pages separate allows us to preserve
    page metadata for every chunk.
    """

    pages = []

    with pdfplumber.open(
        io.BytesIO(content)
    ) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):

            page_text = page.extract_text()

            # extract_text() can return None
            if not page_text:
                continue

            page_text = page_text.strip()

            if not page_text:
                continue

            pages.append(
                {
                    "page": page_number,
                    "text": page_text,
                }
            )

    return pages


# ============================================================
# Create PDF Chunks
# ============================================================

def create_pdf_chunks(
    content: bytes,
) -> list[dict]:
    """
    Extract PDF pages and split each page into chunks.

    Chunks do not cross page boundaries.
    """

    pages = extract_pdf_pages(content)

    chunks = []

    global_chunk_index = 0

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = chunk_text(page_text)

        for page_chunk_index, chunk in enumerate(
            page_chunks
        ):

            chunks.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "chunk_index": global_chunk_index,
                    "page_chunk_index": page_chunk_index,
                }
            )

            global_chunk_index += 1

    return chunks


# ============================================================
# Create Text File Chunks
# ============================================================

def create_text_chunks(
    content: bytes,
) -> list[dict]:
    """
    Decode a UTF-8 text file and split it into chunks.
    """

    text = content.decode("utf-8")

    chunks = chunk_text(text)

    return [
        {
            "text": chunk,
            "chunk_index": index,
        }
        for index, chunk in enumerate(chunks)
    ]


# ============================================================
# /store
#
# Store user memory
# ============================================================

@app.post("/store")
def add_memory(
    request: MemoryRequest,
):

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
#
# Store PDF/text as knowledge
# ============================================================

@app.post("/add_memory_file_upload")
async def add_memory_file_upload(
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    content = await file.read()

    filename = file.filename or ""

    # --------------------------------------------------------
    # Extract and chunk
    # --------------------------------------------------------

    if filename.lower().endswith(".pdf"):

        chunks = create_pdf_chunks(
            content
        )

    else:

        try:

            chunks = create_text_chunks(
                content
            )

        except UnicodeDecodeError:

            return {
                "error": (
                    "File must be a UTF-8 text file "
                    "or PDF"
                )
            }

    # --------------------------------------------------------
    # Empty file check
    # --------------------------------------------------------

    if not chunks:

        return {
            "error": (
                "File is empty or contains "
                "no extractable text"
            )
        }

    # --------------------------------------------------------
    # Store knowledge chunks
    # --------------------------------------------------------

    stored_ids = []

    for chunk in chunks:

        metadata = {
            "file_name": filename,
            "chunk_index": chunk["chunk_index"],
        }

        # PDF-specific metadata
        if "page" in chunk:

            metadata["page"] = chunk["page"]

            metadata["page_chunk_index"] = (
                chunk["page_chunk_index"]
            )

        point_id = store_knowledge(
            text=chunk["text"],
            metadata=metadata,
        )

        stored_ids.append(point_id)

    return {
        "status": "stored",
        "collection": KNOWLEDGE_COLLECTION,
        "file_name": filename,
        "chunks_stored": len(chunks),
        "ids": stored_ids,
    }


# ============================================================
# /search_memory
#
# Search user memory only
# ============================================================

@app.post("/search_memory")
def search_memory_endpoint(
    request: SearchRequest,
):

    results = search_memory(
        query=request.query,
        limit=request.limit,
    )

    return {
        "collection": MEMORY_COLLECTION,
        "results": [
            {
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# /search_knowledge
#
# Search document knowledge only
# ============================================================

@app.post("/search_knowledge")
def search_knowledge_endpoint(
    request: SearchRequest,
):

    results = search_knowledge(
        query=request.query,
        limit=request.limit,
    )

    return {
        "collection": KNOWLEDGE_COLLECTION,
        "results": [
            {
                "text": point.payload.get("text"),
                "file_name": point.payload.get(
                    "file_name"
                ),
                "page": point.payload.get("page"),
                "chunk_index": point.payload.get(
                    "chunk_index"
                ),
                "page_chunk_index": point.payload.get(
                    "page_chunk_index"
                ),
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# /chat
#
# Phase 1 + Phase 2:
# Chat still searches MEMORY only.
#
# Memory + Knowledge combined retrieval
# will be implemented later.
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
):

    # --------------------------------------------------------
    # Retrieve memory
    # --------------------------------------------------------

    results = search_memory(
        query=request.query,
        limit=request.top_k,
    )

    # --------------------------------------------------------
    # Build memory context
    # --------------------------------------------------------

    context_parts = []

    for point in results:

        part = point.payload.get(
            "text",
            "",
        )

        context_parts.append(part)

    context = (
        "\n---\n".join(context_parts)
        if context_parts
        else "No relevant memory found."
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use the following relevant user memory
to answer the user's question.

--- Memory context ---
{context}
--- End of memory context ---

Question: {request.query}

Instructions:
- Use memory if it helps.
- If memory is not relevant, answer based on general knowledge.
- Keep your answer concise and natural.
"""

    # --------------------------------------------------------
    # Call Ollama
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
        "context_used": context,
    }
