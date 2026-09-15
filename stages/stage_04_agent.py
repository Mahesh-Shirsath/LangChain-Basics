"""
Stage 4 — Tools / agents

WHAT TO BUILD HERE (no code yet — implement yourself):

1. Define at least two tools (prefer @tool from langchain):
   a) calculator(expression: str) -> str
      - Evaluate simple arithmetic safely (restrict to math; avoid raw unrestricted eval
        of arbitrary Python if you can use a safer approach).
   b) lookup_order_status(order_id: str) -> str
      - Fake in-memory dict of order IDs → status strings (e.g. ORD-1001 → shipped).
2. Build an agent with create_agent from langchain.agents (LangChain 1.x):
   - pass model (ChatOpenAI), tools, system_prompt
   - for multi-turn memory, pass checkpointer=InMemorySaver() and invoke with a thread_id
3. If create_agent is unavailable on your installed version, fall back to
   langgraph.prebuilt.create_react_agent — but prefer create_agent and note it in comments.
4. REPL that exercises:
   - a math question that must use the calculator
   - an order-status question that must use lookup_order_status
   - a normal chat question that needs no tool
5. Optionally print/stream intermediate agent/tool steps so the loop is visible.

LEARNING GOALS:
- Tool definitions and tool-calling
- Agentic loop (model ↔ tools)
- Modern create_agent / LangGraph checkpointer pattern

SUCCESS CHECK:
- “What’s 17 * 24?” uses the calculator.
- “Status of ORD-1001?” uses the order tool.
- Casual chat still works without forced tool use.

OUT OF SCOPE FOR THIS FILE:
- Real web search API (optional later)
- Full RAG pipeline (that’s Stage 3; you may mention docs but don’t re-implement RAG here)
"""

# TODO: implement Stage 4 below — @tool helpers, create_agent, checkpointer, REPL.

import os
import ast
import operator as op

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


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
# 2. LLM
# ============================================================

model = ChatOpenAI(
    model=model_name,
    api_key=hf_token,
    base_url="https://router.huggingface.co/v1",
    temperature=0,
)


# ============================================================
# 3. SAFE CALCULATOR
# ============================================================

# Allowed arithmetic operators
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}


def _safe_eval(node):
    """
    Recursively evaluate only safe arithmetic expressions.
    """

    if isinstance(node, ast.Constant):

        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):

        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        # Prevent huge exponent calculations
        if operator_type is ast.Pow and abs(right) > 10:
            raise ValueError("Exponent is too large.")

        return _ALLOWED_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):

        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unary operator not allowed.")

        operand = _safe_eval(node.operand)

        return _ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Invalid arithmetic expression.")


@tool
def calculator(expression: str) -> str:
    """
    Calculate a simple arithmetic expression.

    Use this tool for mathematical calculations involving
    numbers and arithmetic operators such as +, -, *, /, %, and **.
    """

    try:
        tree = ast.parse(expression, mode="eval")

        result = _safe_eval(tree.body)

        return str(result)

    except Exception as e:
        return f"Calculator error: {e}"


# ============================================================
# 4. FAKE ORDER DATABASE
# ============================================================

ORDERS = {
    "ORD-1001": "shipped",
    "ORD-1002": "processing",
    "ORD-1003": "delivered",
    "ORD-1004": "cancelled",
}


@tool
def lookup_order_status(order_id: str) -> str:
    """
    Look up the status of an order using its order ID.
    """

    order_id = order_id.strip().upper()

    status = ORDERS.get(order_id)

    if status is None:
        return f"Order {order_id} was not found."

    return f"Order {order_id} is currently {status}."


# ============================================================
# 5. TOOLS
# ============================================================

tools = [
    calculator,
    lookup_order_status,
]


# ============================================================
# 6. CHECKPOINTER
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# 7. AGENT
# ============================================================

agent = create_agent(
   model=model,
   tools=tools,
   system_prompt="""
   You are a helpful technical assistant.

   You have access to two tools:

   1. calculator
      - Use it for arithmetic calculations.
      - Always use the calculator for mathematical calculations
      instead of calculating mentally.

   2. lookup_order_status
      - Use it when the user asks about an order status.
      - Do not invent order statuses.

   For normal conversational questions that do not require a tool,
   answer directly.

   Be concise and clear.
   """,
   checkpointer=checkpointer,
)


# ============================================================
# 8. THREAD CONFIGURATION
# ============================================================

config = {
    "configurable": {
        "thread_id": "stage4-demo"
    }
}


# ============================================================
# 9. REPL
# ============================================================

print("\n========================================")
print("       Stage 4 — Agent + Tools")
print("========================================")
print("Try:")
print("  What's 17 * 24?")
print("  Status of ORD-1001?")
print("  What is Python?")
print("Type 'exit' to stop.")
print("========================================\n")


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        },
        config=config,
    )

    final_message = result["messages"][-1]

    print("Assistant:", final_message.content)
    print()