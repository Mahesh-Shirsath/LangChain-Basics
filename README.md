# LangChain learning stages (OpenAI)

Progressive scripts under `stages/`. Implement one stage at a time; each file’s comments describe what to build.

## Setup

1. Create and activate a virtualenv in this folder.
2. `pip install -r requirements.txt`
3. Copy `.env.example` → `.env` and set `OPENAI_API_KEY`.
4. Run stages in order: `python stages/stage_01_basic_llm.py`, then 02, …

## Layout

| Path | Purpose |
|------|---------|
| `stages/stage_01_basic_llm.py` | Prompt → LLM → parser (LCEL), streaming |
| `stages/stage_02_memory.py` | Chat history via `RunnableWithMessageHistory` |
| `stages/stage_03_rag.py` | Load / split / embed / Chroma / retrieve / generate |
| `stages/stage_04_agent.py` | Tools + `create_agent` |
| `stages/stage_05_structured_tracing.py` | Pydantic structured output + LangSmith |
| `data/docs/` | Sample markdown corpus for Stage 3 RAG |
| `chroma_db/` | Created at runtime by Stage 3 (gitignored) |

## Notes

- Provider: OpenAI (`ChatOpenAI`).
- Vector store: Chroma, persisted under `chroma_db/`.
- Stage 4 tools: calculator + fake order-status lookup (no extra APIs).
- Do not implement later stages inside earlier files; keep each stage runnable on its own.
