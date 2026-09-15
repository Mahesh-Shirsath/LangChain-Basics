"""
Stage 3 — Retrieval-Augmented Generation (RAG)

WHAT TO BUILD HERE (no code yet — implement yourself):

1. Load documents from ../data/docs/ (markdown). Use DirectoryLoader / TextLoader
   (or equivalent) so each file becomes Document(s).
2. Split with RecursiveCharacterTextSplitter (e.g. chunk_size=500, chunk_overlap=50).
3. Embed with OpenAIEmbeddings and store in Chroma:
   - persist_directory pointing at ../chroma_db/ (or project-root chroma_db/)
4. Build a retriever (e.g. as_retriever with k=3).
5. Wire a RAG LCEL chain roughly like:
   - retrieve context for the question
   - format docs into a string
   - prompt that says: answer ONLY from context; include {context} and {question}
   - llm | StrOutputParser()
   Pattern sketch (implement for real):
     {"context": retriever | format_docs, "question": RunnablePassthrough()}
     | prompt | llm | parser
6. REPL: ask questions answerable from data/docs; also try a question outside the corpus
   to observe grounding / “not in docs” behavior.

LEARNING GOALS:
- Document loaders
- RecursiveCharacterTextSplitter
- Embeddings + Chroma
- Retriever wired into LCEL

SUCCESS CHECK:
- A detail that exists only in one of the markdown files is answered correctly.
- Unrelated trivia is not invented (or the model says it isn’t in the docs).

PREREQUISITE CONTENT:
- Fill in data/docs/*.md before running (see comments in those files).

OUT OF SCOPE FOR THIS FILE:
- Agents / tools (Stage 4)
- Structured output / LangSmith (Stage 5)
"""

# TODO: implement Stage 3 below — load, split, embed, Chroma, retriever, RAG chain, REPL.

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("HF_MODEL")

if not hf_token:
    raise ValueError("HF_TOKEN is not set in .env")

if not model_name:
    raise ValueError("HF_MODEL is not set in .env")


# ============================================================
# 2. PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

docs_path = project_root / "data" / "docs"
chroma_path = project_root / "chroma_db"


# ============================================================
# 3. LOAD DOCUMENTS
# ============================================================

loader = DirectoryLoader(
   str(docs_path),
   glob="*.md",
   loader_cls=TextLoader,
   loader_kwargs={"encoding": "utf-8"},
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")


# ============================================================
# 4. SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=500,
   chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# ============================================================
# 5. CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
   model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 6. STORE EMBEDDINGS IN CHROMA
# ============================================================

chroma = Chroma.from_documents(
   chunks,
   embeddings,
   persist_directory=str(chroma_path),
)

print("Documents stored in Chroma")


# ============================================================
# 7. CREATE RETRIEVER
# ============================================================

retriever = chroma.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 8. PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
   """
      Answer the question using ONLY the provided context.

      If the answer is not present in the context, say:
      "I don't know based on the provided documents."

      Context:
      {context}

      Question:
      {question}

      Answer:
   """
   )


# ============================================================
# 9. LLM
# ============================================================

llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    api_key=hf_token,
    base_url="https://router.huggingface.co/v1",
)


# ============================================================
# 10. OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# 11. FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# 12. RAG LCEL CHAIN
# ============================================================

rag_chain = (
   {
      "context": retriever | format_docs,
      "question": RunnablePassthrough(),
   }
   | prompt
   | llm
   | parser
)


# ============================================================
# 13. REPL
# ============================================================

print("\n================ RAG CHATBOT ================")
print("Ask questions about your documents.")
print("Type 'exit' to stop.\n")

# print("\n========== RETRIEVER DEBUG ==========\n")

# docs = retriever.invoke("products overview")

# print(f"Retrieved {len(docs)} chunks\n")

# for i, doc in enumerate(docs, start=1):
#     print(f"--- CHUNK {i} ---")
#     print("SOURCE:", doc.metadata.get("source"))
#     print(doc.page_content)
#     print()


while True:

   question = input("You: ")

   if question.lower() in ["exit", "quit"]:
      break

   print("Assistant: ", end="", flush=True)

   for chunk in rag_chain.stream(question):
      print(chunk, end="", flush=True)

   print("\n")