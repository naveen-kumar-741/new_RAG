from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import ollama
import pdfplumber
import io


app = FastAPI()


# ============================================================
# Configuration
# ============================================================

MEMORY_COLLECTION = "memory"
KNOWLEDGE_COLLECTION = "knowledge"

VECTOR_SIZE = 384


# ============================================================
# Qdrant
# ============================================================

# Initialize in-memory Qdrant
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

def store_memory(text: str, metadata: dict | None = None):
    """
    Store a user memory in the memory collection.
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

def store_knowledge(text: str, metadata: dict | None = None):
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

def search_memory(query: str, limit: int = 3):
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

def search_knowledge(query: str, limit: int = 3):
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
# /store
# Store User Memory
# ============================================================

@app.post("/store")
def add_memory(request: MemoryRequest):

    point_id = store_memory(
        text=request.text,
        metadata={
            "type": "memory",
        },
    )

    return {
        "status": "stored",
        "collection": MEMORY_COLLECTION,
        "id": point_id,
    }


# ============================================================
# /add_memory_file_upload
# Store Document Knowledge
# ============================================================

@app.post("/add_memory_file_upload")
async def add_memory_file_upload(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    content = await file.read()

    filename = file.filename or ""

    text = ""

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if filename.lower().endswith(".pdf"):

        with pdfplumber.open(io.BytesIO(content)) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    # --------------------------------------------------------
    # Text file
    # --------------------------------------------------------

    else:

        try:
            text = content.decode("utf-8")

        except UnicodeDecodeError:

            return {
                "error": "File must be a text or PDF file"
            }

    # --------------------------------------------------------
    # Empty file check
    # --------------------------------------------------------

    if not text.strip():

        return {
            "error": "File is empty"
        }

    # --------------------------------------------------------
    # Chunking
    #
    # NOTE:
    # We are intentionally keeping the existing simple
    # chunking for Phase 1.
    #
    # Chunking improvements belong to Phase 2.
    # --------------------------------------------------------

    chunks = [
        text[i:i + 1000]
        for i in range(0, len(text), 1000)
    ]

    stored_ids = []

    # --------------------------------------------------------
    # Store chunks in KNOWLEDGE collection
    # --------------------------------------------------------

    for chunk_index, chunk in enumerate(chunks):

        point_id = store_knowledge(
            text=chunk,
            metadata={
                "file_name": filename,
                "chunk_index": chunk_index,
            },
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
# Search User Memories Only
# ============================================================

@app.post("/search_memory")
def search_memory_endpoint(request: SearchRequest):

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
# Search Documents Only
# ============================================================

@app.post("/search_knowledge")
def search_knowledge_endpoint(request: SearchRequest):

    results = search_knowledge(
        query=request.query,
        limit=request.limit,
    )

    return {
        "collection": KNOWLEDGE_COLLECTION,
        "results": [
            {
                "text": point.payload.get("text"),
                "file_name": point.payload.get("file_name"),
                "chunk_index": point.payload.get("chunk_index"),
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# /chat
#
# Phase 1:
# Keep chat behavior mostly unchanged.
#
# It currently searches MEMORY only.
#
# Combining MEMORY + KNOWLEDGE belongs to Phase 4.
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    results = search_memory(
        query=request.query,
        limit=request.top_k,
    )

    # --------------------------------------------------------
    # Build memory context
    # --------------------------------------------------------

    context_parts = []

    for point in results:

        part = point.payload.get("text", "")

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

Use the following relevant user memory to answer
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