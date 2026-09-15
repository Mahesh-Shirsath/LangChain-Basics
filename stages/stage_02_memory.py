"""
Stage 2 — Add memory (conversational chatbot)

WHAT TO BUILD HERE (no code yet — implement yourself):

1. Reuse Stage 1 ideas: ChatOpenAI + ChatPromptTemplate + StrOutputParser.
2. Extend the prompt with MessagesPlaceholder("history") so prior turns are injected.
3. Keep an in-memory session store (e.g. dict of session_id → ChatMessageHistory
   or InMemoryChatMessageHistory).
4. Wrap the LCEL chain with RunnableWithMessageHistory:
   - get_session_history callback
   - input_messages_key / history_messages_key wired to your prompt variables
5. Run a simple REPL loop:
   - read user input
   - invoke with config={"configurable": {"session_id": "demo"}} (or similar)
   - print the assistant reply
6. Optional: after a few turns, print how many messages are in history.

LEARNING GOALS:
- MessagesPlaceholder
- RunnableWithMessageHistory and session_id
- Multi-turn conversation without manually resending history

SUCCESS CHECK:
- Say “My name is Ada”, then ask “What’s my name?” — model recalls Ada.

OUT OF SCOPE FOR THIS FILE:
- Vector store / RAG (Stage 3)
- Tools / agents (Stage 4)
"""

# TODO: implement Stage 2 below — history placeholder, session store, RWMH, REPL.
import os 

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# ============================================================
# 1. Load the environment variables
# ============================================================

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
   raise ValueError("HF_TOKEN is not set in .env")

model_name = os.getenv("HF_MODEL")
if not model_name:
   raise ValueError("HF_MODEL is not set in .env")

# ============================================================
# 2. Create the LLM
# ============================================================

llm = ChatOpenAI(
    model=model_name,
    api_key=hf_token,
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=256,
)

# ============================================================
# 3. Create the ChatPromptTemplate
# ============================================================

prompt = ChatPromptTemplate.from_messages([
   (
      "system",
      "You are a helpful technical assistant. "
   ),
   MessagesPlaceholder("history"),
   ("user", "{input}"),
])

# ============================================================
# 4. Create the StrOutputParser
# ============================================================

output_parser = StrOutputParser()

# ============================================================
# 5. Create the session store
# ============================================================ 

store: dict[str, InMemoryChatMessageHistory] = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
   """Get the session history from the store
   
   Args:
      session_id: The session ID
   Returns:
      The session history
   """
   if session_id not in store:
      store[session_id] = InMemoryChatMessageHistory()

   return store[session_id]

# ============================================================
# 6. Create the RunnableWithMessageHistory
# ============================================================

runnable = prompt | llm | output_parser
runnable_with_message_history = RunnableWithMessageHistory(
   runnable,
   get_session_history=get_session_history,
   input_messages_key="input",
   history_messages_key="history",
)

# ============================================================
# 7. Create the REPL loop
# ============================================================
print("Conversational Chatbot")
print("Type 'exit' or 'quit' to stop.\n")


while True:

   user_input = input("You: ")

   if user_input.lower() in ["exit", "quit"]:
      break

   response = runnable_with_message_history.invoke(
      {
         "input": user_input
      },
      config={
         "configurable": {
            "session_id": "conversation"
         }
      }
   )

   print("Assistant:", response)

   # Optional history size
   history = store["conversation"]

   print(f"[Messages in history: {len(history.messages)}]\n")