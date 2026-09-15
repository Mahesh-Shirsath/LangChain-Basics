"""
Stage 1 — Basic LLM call

WHAT TO BUILD HERE (no code yet — implement yourself):

1. Load environment variables from .env (OPENAI_API_KEY, OPENAI_MODEL).
2. Create a ChatOpenAI chat model (langchain_openai).
3. Build a ChatPromptTemplate with message roles:
   - system: short assistant persona / instructions
   - human: template with a variable (e.g. {topic} or {question})
4. Compose an LCEL chain: prompt | llm | StrOutputParser()
5. Demonstrate chain.invoke(...) for a one-shot answer.
6. Demonstrate chain.stream(...) and print token/chunk output so you feel streaming.

LEARNING GOALS:
- LCEL pipe syntax
- ChatPromptTemplate and system vs human roles
- Output parsers
- Streaming vs invoke

SUCCESS CHECK:
- Script prints a full answer once.
- Script then prints a second answer streamed chunk-by-chunk.

OUT OF SCOPE FOR THIS FILE:
- Chat history / memory (Stage 2)
- Retrieval / RAG (Stage 3)
- Tools / agents (Stage 4)
"""

# TODO: implement Stage 1 below — imports, dotenv, prompt | llm | parser, invoke + stream.

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("HF_MODEL")


if not hf_token:
    raise ValueError("HF_TOKEN is not set in .env")

if not model_name:
    raise ValueError("HF_MODEL is not set in .env")


# ============================================================
# 2. Create the Hugging Face LLM
# ============================================================

llm = ChatOpenAI(
    model=model_name,
    api_key=hf_token,
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=256,
)


# ============================================================
# 3. Create ChatPromptTemplate
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful technical assistant. "
        "Explain technical concepts clearly and simply."
    ),
    (
        "human",
        "Explain {topic} in simple terms."
    )
])


# ============================================================
# 4. Create Output Parser
# ============================================================

parser = StrOutputParser()


# ============================================================
# 5. Create LCEL Chain
# ============================================================

chain = prompt | llm | parser


# ============================================================
# 6. Invoke - complete response
# ============================================================

print("\n================ INVOKE ================\n")

response = chain.invoke({
    "topic": "RAG"
})

print(response)


# ============================================================
# 7. Stream - chunk by chunk response
# ============================================================

print("\n================ STREAM ================\n")

for chunk in chain.stream({
    "topic": "FastAPI"
}):
    print(chunk, end="", flush=True)

print("\n")