from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import ollama
import pdfplumber
import io
import json
from datetime import datetime, timezone

app = FastAPI()


# ============================================================
# Configuration
# ============================================================

MEMORY_COLLECTION = "memory"
KNOWLEDGE_COLLECTION = "knowledge"

VECTOR_SIZE = 384

# Used when retrieving information for answering questions
DEFAULT_SCORE_THRESHOLD = 0.3

# Used only when checking whether a new memory is a duplicate
MEMORY_DUPLICATE_THRESHOLD = 0.75

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "phi3:mini"


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

model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# Request Models
# ============================================================


class MemoryRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 3
    score_threshold: float = DEFAULT_SCORE_THRESHOLD


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    score_threshold: float = DEFAULT_SCORE_THRESHOLD


# ============================================================
# Memory Analyzer Model
# ============================================================


class MemoryCandidate(BaseModel):
    remember: bool
    type: str | None = None
    text: str | None = None
    importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# Utility Functions
# ============================================================


def create_embedding(text: str):
    """
    Convert text into an embedding vector.
    """
    return model.encode(text).tolist()


def current_timestamp():
    """
    Return current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Memory Storage
# ============================================================


def store_memory(
    text: str,
    memory_type: str = "fact",
    importance: float = 0.5,
    source: str = "manual",
):
    """
    Store a memory in the memory collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    timestamp = current_timestamp()

    payload = {
        "text": text,
        "type": memory_type,
        "importance": importance,
        "source": source,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "active",
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
    file_name: str,
    chunk_index: int,
    page: int | None = None,
):
    """
    Store a knowledge chunk in the knowledge collection.
    """

    embedding = create_embedding(text)

    point_id = str(uuid.uuid4())

    payload = {
        "text": text,
        "type": "knowledge",
        "file_name": file_name,
        "page": page,
        "chunk_index": chunk_index,
        "created_at": current_timestamp(),
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
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):
    """
    Search only the memory collection.
    """

    query_embedding = create_embedding(query)

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=score_threshold,
    )

    return results.points


# ============================================================
# Knowledge Search
# ============================================================


def search_knowledge(
    query: str,
    limit: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):
    """
    Search only the knowledge collection.
    """

    query_embedding = create_embedding(query)

    results = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=score_threshold,
    )

    return results.points


# ============================================================
# Duplicate Memory Detection
# ============================================================


def find_duplicate_memory(
    text: str,
    duplicate_threshold: float = MEMORY_DUPLICATE_THRESHOLD,
):
    """
    Search existing memories to determine whether the supplied
    memory is semantically similar to an existing memory.

    This threshold is intentionally separate from the normal
    retrieval threshold.
    """

    embedding = create_embedding(text)

    results = qdrant.query_points(
        collection_name=MEMORY_COLLECTION,
        query=embedding,
        limit=1,
        score_threshold=duplicate_threshold,
    )

    if not results.points:
        return {
            "is_duplicate": False,
            "existing_memory_id": None,
            "existing_text": None,
            "score": None,
        }

    point = results.points[0]

    return {
        "is_duplicate": True,
        "existing_memory_id": str(point.id),
        "existing_text": point.payload.get("text"),
        "score": float(point.score),
    }


# ============================================================
# Store Memory If Not Duplicate
# ============================================================


def store_memory_if_not_duplicate(
    text: str,
    memory_type: str,
    importance: float,
    source: str,
):
    """
    Check for an existing similar memory before storing.

    Returns information describing whether the memory was
    created or skipped because it was a duplicate.
    """

    duplicate_result = find_duplicate_memory(text)

    if duplicate_result["is_duplicate"]:

        return {
            "memory_stored": False,
            "memory_duplicate": True,
            "memory_id": None,
            "existing_memory_id": duplicate_result["existing_memory_id"],
            "existing_memory_text": duplicate_result["existing_text"],
            "duplicate_score": duplicate_result["score"],
        }

    memory_id = store_memory(
        text=text,
        memory_type=memory_type,
        importance=importance,
        source=source,
    )

    return {
        "memory_stored": True,
        "memory_duplicate": False,
        "memory_id": memory_id,
        "existing_memory_id": None,
        "existing_memory_text": None,
        "duplicate_score": None,
    }


# ============================================================
# Manual Memory Endpoint
# ============================================================


@app.post("/store")
def add_memory(request: MemoryRequest):

    result = store_memory_if_not_duplicate(
        text=request.text,
        memory_type="memory",
        importance=0.5,
        source="manual",
    )

    return {
        "status": ("duplicate" if result["memory_duplicate"] else "stored"),
        "collection": MEMORY_COLLECTION,
        **result,
    }


# ============================================================
# Knowledge File Upload
# ============================================================


@app.post("/add_memory_file_upload")
async def add_memory_file_upload(file: UploadFile = File(...)):

    content = await file.read()

    file_name = file.filename or "unknown"

    text = ""

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if file_name.lower().endswith(".pdf"):

        with pdfplumber.open(io.BytesIO(content)) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    else:

        try:
            text = content.decode("utf-8")

        except UnicodeDecodeError:

            return {"error": "File must be a text or PDF file"}

    # --------------------------------------------------------
    # Empty File
    # --------------------------------------------------------

    if not text.strip():

        return {"error": "File is empty"}

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]

    stored_ids = []

    for chunk_index, chunk in enumerate(chunks):

        point_id = store_knowledge(
            text=chunk,
            file_name=file_name,
            chunk_index=chunk_index,
        )

        stored_ids.append(point_id)

    return {
        "status": "stored",
        "collection": KNOWLEDGE_COLLECTION,
        "file_name": file_name,
        "chunks_stored": len(chunks),
        "ids": stored_ids,
    }


# ============================================================
# Search Memory Endpoint
# ============================================================


@app.post("/search_memory")
def search_memory_endpoint(request: SearchRequest):

    results = search_memory(
        query=request.query,
        limit=request.limit,
        score_threshold=request.score_threshold,
    )

    return {
        "collection": MEMORY_COLLECTION,
        "score_threshold": request.score_threshold,
        "results": [
            {
                "id": str(point.id),
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "importance": point.payload.get("importance"),
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# Search Knowledge Endpoint
# ============================================================


@app.post("/search_knowledge")
def search_knowledge_endpoint(request: SearchRequest):

    results = search_knowledge(
        query=request.query,
        limit=request.limit,
        score_threshold=request.score_threshold,
    )

    return {
        "collection": KNOWLEDGE_COLLECTION,
        "score_threshold": request.score_threshold,
        "results": [
            {
                "id": str(point.id),
                "text": point.payload.get("text"),
                "type": point.payload.get("type"),
                "file_name": point.payload.get("file_name"),
                "page": point.payload.get("page"),
                "chunk_index": point.payload.get("chunk_index"),
                "score": float(point.score),
            }
            for point in results
        ],
    }


# ============================================================
# Memory Analyzer
# ============================================================


def analyze_memory(user_message: str) -> MemoryCandidate:
    """
    Analyze a user message and determine whether it contains
    durable information worth remembering.
    """

    prompt = f"""
You are a memory extraction system.

Your ONLY job is to analyze the user's message and determine
whether it contains durable information about the user that
would be useful in future conversations.

DO NOT answer the user's question.

DO NOT provide explanations.

Return ONLY JSON.

------------------------------------------------------------
WHEN TO REMEMBER
------------------------------------------------------------

Remember the message if it contains:

1. A user preference.

Examples:

"I prefer TypeScript."
"I like concise explanations."
"I prefer dark mode."

2. A stable fact about the user.

Examples:

"I mainly work with React."
"I use VS Code for development."
"I work mostly on frontend projects."

3. A persistent instruction.

Examples:

"From now on, always explain programming concepts with examples."
"Always use TypeScript in your examples."
"Keep your answers concise."
"Don't use overly complicated explanations."

4. A change to an existing preference or workflow.

Examples:

"I've switched to TypeScript."
"I no longer use JavaScript."
"I now use VS Code."

------------------------------------------------------------
WHEN NOT TO REMEMBER
------------------------------------------------------------

Do NOT remember:

- General knowledge questions
- One-time questions
- Greetings
- Thanks
- Temporary requests
- Requests to explain a topic
- Questions about unrelated facts
- Information that does not describe a durable user preference,
  fact, or instruction

Examples:

"What is React?"

"What is the capital of France?"

"How does Python work?"

"Thanks!"

"Can you explain recursion?"

------------------------------------------------------------
IMPORTANT NORMALIZATION RULE
------------------------------------------------------------

Convert the user's wording into a concise, reusable statement.

For technology preferences, prefer wording such as:

"User prefers TypeScript for projects."

instead of:

"User has switched exclusively to using TypeScript for
their projects."

For development facts:

"User mainly works with React."

For editor preferences:

"User uses VS Code for development."

For instructions:

"User prefers programming concepts to be explained with
simple examples."

------------------------------------------------------------
MEMORY TYPES
------------------------------------------------------------

Allowed types:

- preference
- fact
- instruction

------------------------------------------------------------
IMPORTANCE
------------------------------------------------------------

Use a value between 0 and 1.

Examples:

0.9 = very important persistent preference/instruction

0.7 = useful long-term fact

0.5 = relatively minor preference

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

If the message should be remembered:

{{
    "remember": true,
    "type": "preference",
    "text": "User prefers TypeScript for projects.",
    "importance": 0.8
}}

If the message should be ignored:

{{
    "remember": false,
    "type": null,
    "text": null,
    "importance": 0
}}

------------------------------------------------------------

USER MESSAGE:

{user_message}
"""

    try:

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format="json",
        )

        raw_content = response["message"]["content"].strip()

        parsed = json.loads(raw_content)

        candidate = MemoryCandidate(**parsed)

    except Exception:

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    allowed_types = {
        "preference",
        "fact",
        "instruction",
    }

    if not candidate.remember:

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    if candidate.type not in allowed_types:

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    if not candidate.text or not candidate.text.strip():

        return MemoryCandidate(
            remember=False,
            type=None,
            text=None,
            importance=0.0,
        )

    importance = max(
        0.0,
        min(1.0, candidate.importance),
    )

    return MemoryCandidate(
        remember=True,
        type=candidate.type,
        text=candidate.text.strip(),
        importance=importance,
    )


# ============================================================
# Chat Endpoint
# ============================================================


@app.post("/chat")
def chat(request: ChatRequest):

    # --------------------------------------------------------
    # 1. Retrieve Memory
    # --------------------------------------------------------

    memory_results = search_memory(
        query=request.query,
        limit=request.top_k,
        score_threshold=request.score_threshold,
    )

    # --------------------------------------------------------
    # 2. Retrieve Knowledge
    # --------------------------------------------------------

    knowledge_results = search_knowledge(
        query=request.query,
        limit=request.top_k,
        score_threshold=request.score_threshold,
    )

    # --------------------------------------------------------
    # 3. Build Memory Context
    # --------------------------------------------------------

    memory_context_parts = []

    for point in memory_results:

        part = f"[Score: {point.score:.3f}]\n" f"{point.payload.get('text', '')}"

        memory_context_parts.append(part)

    if memory_context_parts:

        memory_context = "\n---\n".join(memory_context_parts)

    else:

        memory_context = "No relevant memory found."

    # --------------------------------------------------------
    # 4. Build Knowledge Context
    # --------------------------------------------------------

    knowledge_context_parts = []

    for point in knowledge_results:

        part = f"[Score: {point.score:.3f}]\n"

        if point.payload.get("file_name"):

            part += f"File: " f"{point.payload['file_name']}"

        if point.payload.get("page") is not None:

            part += f" | Page: " f"{point.payload['page']}"

        if point.payload.get("chunk_index") is not None:

            part += f" | Chunk: " f"{point.payload['chunk_index']}"

        part += f"\n{point.payload.get('text', '')}"

        knowledge_context_parts.append(part)

    if knowledge_context_parts:

        knowledge_context = "\n---\n".join(knowledge_context_parts)

    else:

        knowledge_context = "No relevant knowledge found."

    # --------------------------------------------------------
    # 5. Combined Context
    # --------------------------------------------------------

    context = (
        "\n--- User Memory ---\n"
        f"{memory_context}\n"
        "\n--- Knowledge ---\n"
        f"{knowledge_context}\n"
    )

    # --------------------------------------------------------
    # 6. Main LLM Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use the provided memory and knowledge when they
are relevant to the user's question.

--- User Memory ---
{memory_context}

--- Knowledge ---
{knowledge_context}

--- End Context ---

Question:
{request.query}

Instructions:

- Use relevant memory when answering.
- Use relevant knowledge when answering.
- Do not mention the retrieval system.
- Do not invent information.
- If the context does not contain the answer,
  use your general knowledge when appropriate.
- Keep the answer concise and natural.
"""

    # --------------------------------------------------------
    # 7. Generate Answer
    # --------------------------------------------------------

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response["message"]["content"]

    # --------------------------------------------------------
    # 8. Analyze User Message
    # --------------------------------------------------------

    memory_candidate = analyze_memory(request.query)

    # --------------------------------------------------------
    # 9. Duplicate Check + Store
    # --------------------------------------------------------

    memory_result = {
        "memory_stored": False,
        "memory_duplicate": False,
        "memory_id": None,
        "existing_memory_id": None,
        "existing_memory_text": None,
        "duplicate_score": None,
    }

    if memory_candidate.remember:

        memory_result = store_memory_if_not_duplicate(
            text=memory_candidate.text,
            memory_type=memory_candidate.type,
            importance=memory_candidate.importance,
            source="conversation",
        )

    # --------------------------------------------------------
    # 10. Return Response
    # --------------------------------------------------------

    return {
        "response": answer,
        "score_threshold": request.score_threshold,
        "memory_results": len(memory_results),
        "knowledge_results": len(knowledge_results),
        "context_used": context,
        "memory_analysis": {
            "remember": memory_candidate.remember,
            "type": memory_candidate.type,
            "text": memory_candidate.text,
            "importance": memory_candidate.importance,
        },
        "memory_stored": memory_result["memory_stored"],
        "memory_duplicate": memory_result["memory_duplicate"],
        "memory_id": memory_result["memory_id"],
        "existing_memory_id": memory_result["existing_memory_id"],
        "existing_memory_text": memory_result["existing_memory_text"],
        "duplicate_score": memory_result["duplicate_score"],
    }
