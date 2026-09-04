from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import ollama
import pdfplumber

app = FastAPI()

# Initialize in-memory Qdrant
qdrant = QdrantClient(":memory:")
qdrant.recreate_collection(
    collection_name="memory",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Models
# -------------------------------
class MemoryRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 3

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3  # number of relevant memories to retrieve

# -------------------------------
# Text memory endpoint
# -------------------------------
@app.post("/store")
def add_memory(request: MemoryRequest):
    embedding = model.encode(request.text).tolist()
    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name="memory",
        points=[PointStruct(id=point_id, vector=embedding, payload={"text": request.text})],
    )
    return {"status": "stored", "id": point_id}

# -------------------------------
# File memory endpoint
# -------------------------------

@app.post("/add_memory_file_upload")
async def add_memory_file_upload(file: UploadFile = File(...)):
    # Read file content
    content = await file.read()
    
    text = ""
    if file.filename.endswith(".pdf"):
        # Extract text from PDF
        import io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    else:
        # Treat as text file
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "File must be a text or PDF file"}
    
    if not text.strip():
        return {"error": "File is empty"}

    # Chunking (optional, recommended for large PDFs)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    stored_ids = []
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        point_id = str(uuid.uuid4())
        payload = {"text": chunk, "file_name": file.filename}
        qdrant.upsert(
            collection_name="memory",
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
        )
        stored_ids.append(point_id)

    return {"status": "stored", "file_name": file.filename, "chunks_stored": len(chunks), "ids": stored_ids}

# -------------------------------
# Search memory endpoint
# -------------------------------
@app.post("/search_memory")
def search_memory(request: SearchRequest):
    query_embedding = model.encode(request.query).tolist()
    results = qdrant.query_points(
        collection_name="memory",
        query=query_embedding,
        limit=request.limit,
    )
    return {
        "results": [
            {
                "text": point.payload.get("text"),
                "file_name": point.payload.get("file_name"),
                "score": float(point.score)
            }
            for point in results.points
        ]
    }

# -------------------------------
# Chat endpoint with top-k RAG
# -------------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    # Encode query
    query_embedding = model.encode(request.query).tolist()
    # Retrieve top-k relevant memories
    results = qdrant.query_points(
        collection_name="memory",
        query=query_embedding,
        limit=request.top_k,
    )

    # Build context
    context_parts = []
    for point in results.points:
        part = point.payload.get("text", "")
        if point.payload.get("file_name"):
            part += f"\nFile: {point.payload['file_name']}"
        context_parts.append(part)
    context = "\n---\n".join(context_parts) if context_parts else "No relevant memory found."

    # Build RAG prompt
    prompt = f"""
You are a helpful AI assistant. Use the following relevant information to answer the user's question.

--- Memory context ---
{context}
--- End of memory context ---

Question: {request.query}

Instructions:
- Use memory if it helps.
- If memory is not relevant, answer based on general knowledge.
- Keep your answer concise and natural.
"""

    # Call LLM
    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"response": response["message"]["content"], "context_used": context}
