"""
Tool Registry - registers and dispatches all agent tools.
"""

from tools.memory import MemoryTool
from tools.document_store import DocumentStoreTool
from tools.web_search import WebSearchTool
from tools.python_exec import PythonExecTool
from tools.skills import SkillsTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}
        self._schemas: list[dict] = []

        # Instantiate all tools
        all_tools = [
            MemoryTool(),
            DocumentStoreTool(),
            WebSearchTool(),
            PythonExecTool(),
            SkillsTool(),
        ]

        for tool in all_tools:
            for schema in tool.schemas():
                name = schema["function"]["name"]
                self._tools[name] = (tool, name)
                self._schemas.append(schema)

    def get_tool_schemas(self) -> list[dict]:
        return self._schemas

    def call(self, name: str, args: dict):
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        tool, method_name = self._tools[name]
        method = getattr(tool, method_name, None)
        if method is None:
            return {"error": f"Tool {name} has no method {method_name}"}
        return method(**args)
