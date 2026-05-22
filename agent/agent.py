"""
IR Agent - Core agent loop with tool use and context management.
"""

import json
import os
import sys
from typing import Any

from openai import OpenAI

from agent.context import ContextManager
from tools.registry import ToolRegistry


SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools for finding and managing information.

You have access to the following core capabilities:
- **web_search**: Search the web for current information
- **memory_write**: Save important facts or notes to persistent memory
- **memory_read**: Retrieve your saved memory/notes
- **memory_search**: Search through memory using keywords
- **document_add**: Add a document/text to your local knowledge base
- **document_search**: Search your local knowledge base using BM25 ranking
- **list_skills**: List available skill files (reusable instructions)
- **read_skill**: Read a specific skill file for instructions on how to do something
- **python_exec**: Execute Python code for calculations, data processing, etc.

## How to use tools effectively
- Before answering complex questions, search your memory and documents first.
- Use web_search for current events or information you don't have.
- Use memory_write to save key facts, user preferences, or important results.
- Use document_search when you have indexed documents and need to find relevant passages.
- Think step by step and use multiple tools when needed.

## Memory and context
Your memory persists between conversations. Always check memory_read at the start if the user references something from before.
"""


class IRAgent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.tools = ToolRegistry()
        self.context = ContextManager()
        self.conversation: list[dict] = []
        self.max_tool_rounds = 10

    def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response."""
        self.conversation.append({"role": "user", "content": user_message})

        # Auto-compact if conversation is getting long
        self.context.maybe_compact(self.conversation)

        for _ in range(self.max_tool_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(),
                tools=self.tools.get_tool_schemas(),
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # Append assistant message (may include tool_calls)
            assistant_entry: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_entry["content"] = msg.content
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.conversation.append(assistant_entry)

            if not msg.tool_calls:
                # Final text response
                return msg.content or ""

            # Execute all tool calls
            for tc in msg.tool_calls:
                result = self._execute_tool(tc.function.name, tc.function.arguments)
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Max tool rounds reached. Please try rephrasing your question."

    def _build_messages(self) -> list[dict]:
        system = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Include compacted summary if available
        summary = self.context.get_summary_message()
        if summary:
            system.append({"role": "system", "content": summary})
        return system + self.conversation

    def _execute_tool(self, name: str, arguments_json: str) -> str:
        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid arguments JSON"})

        print(f"[tool] {name}({args})")
        try:
            result = self.tools.call(name, args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def reset(self):
        """Clear conversation history (keeps memory)."""
        self.conversation = []
