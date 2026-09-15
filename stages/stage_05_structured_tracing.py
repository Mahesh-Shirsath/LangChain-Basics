"""
Stage 5 (stretch) — Structured output + LangSmith tracing

WHAT TO BUILD HERE (no code yet — implement yourself):

1. Define a Pydantic model for the forced shape of the answer, e.g.:
   - answer: str
   - confidence: float
   - sources: list[str]
   (Adjust fields if you like; keep them typed and validated.)
2. Use llm.with_structured_output(YourModel) (optionally inside a small LCEL chain).
3. Invoke on a fixed prompt/question and print the result as a dict / model_dump().
4. Confirm types are correct (confidence is a float, sources is a list, etc.).
5. Hook up LangSmith for visibility:
   - Set in .env: LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT
   - Run this stage (or re-run Stage 1 with tracing on)
   - Open https://smith.langchain.com and inspect the run tree
     (prompt / LLM / structured-output spans)

LEARNING GOALS:
- with_structured_output and Pydantic schemas
- Why tracing matters for debugging chains
- Reading a LangSmith trace

SUCCESS CHECK:
- Console shows a validated Pydantic object (not free-form prose only).
- A matching run appears in LangSmith for your project.

PREREQUISITES:
- OpenAI key still required.
- LangSmith API key only needed for the tracing half; structured output works without it.
"""

# TODO: implement Stage 5 below — Pydantic schema, with_structured_output, LangSmith env check.

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("HF_MODEL")

if not hf_token:
    raise ValueError("HF_TOKEN is not set in .env")

if not model_name:
    raise ValueError("HF_MODEL is not set in .env")

# LangChain/LangSmith pick these up automatically from the environment.
langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langsmith_project = os.getenv("LANGSMITH_PROJECT", "langchain-stages")


# ============================================================
# 2. PYDANTIC OUTPUT MODEL
# ============================================================

class Answer(BaseModel):
    answer: str = Field(
        description="The answer to the user's question.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the answer, between 0 and 1.",
    )
    sources: list[str] = Field(
        description="List of sources used to produce the answer.",
    )


# ============================================================
# 3. LLM
# ============================================================

llm = ChatOpenAI(
    model=model_name,
    api_key=hf_token,
    base_url="https://router.huggingface.co/v1",
    temperature=0,
)


# ============================================================
# 4. STRUCTURED OUTPUT + LCEL CHAIN
# ============================================================

structured_llm = llm.with_structured_output(Answer)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful technical assistant. "
        "Fill every field in the schema. "
        "confidence must be a float between 0 and 1. "
        "sources should be a list of short source names or URLs.",
    ),
    ("human", "{question}"),
])

chain = prompt | structured_llm


# ============================================================
# 5. FIXED QUESTION
# ============================================================

question = (
    "What is RAG in Generative AI? "
    "Explain it briefly and mention the main components involved."
)


# ============================================================
# 6. LANGSMITH STATUS
# ============================================================

print("\n========================================")
print("LangSmith tracing")
print("========================================")
print("LANGSMITH_TRACING:", langsmith_tracing)
print("LANGSMITH_PROJECT:", langsmith_project)

if langsmith_tracing and langsmith_api_key:
    print("Status: ON — inspect the run at https://smith.langchain.com")
elif langsmith_tracing:
    print("Status: tracing is true but LANGSMITH_API_KEY is missing in .env")
else:
    print("Status: OFF — set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env")


# ============================================================
# 7. INVOKE
# ============================================================

result = chain.invoke({"question": question})


# ============================================================
# 8. PRINT PYDANTIC OBJECT
# ============================================================

print("\n========================================")
print("Structured Output")
print("========================================")

print(result)


# ============================================================
# 9. PRINT AS DICT
# ============================================================

print("\n========================================")
print("model_dump()")
print("========================================")

print(result.model_dump())


# ============================================================
# 10. VERIFY TYPES
# ============================================================

print("\n========================================")
print("Type Validation")
print("========================================")

print("answer:", type(result.answer).__name__, isinstance(result.answer, str))
print("confidence:", type(result.confidence).__name__, isinstance(result.confidence, float))
print("sources:", type(result.sources).__name__, isinstance(result.sources, list))

print("\nValues:")
print("answer =", result.answer)
print("confidence =", result.confidence)
print("sources =", result.sources)
