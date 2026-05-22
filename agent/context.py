"""
Context Manager - handles conversation compaction and summary injection.

Compaction: when a conversation grows long, we summarize older turns and
keep only the recent ones + the summary. This prevents context window overflow
while preserving important information (inspired by OpenClaw compaction).
"""

import json
import os
from pathlib import Path


COMPACTION_THRESHOLD = 20   # number of messages before compaction triggers
MESSAGES_TO_KEEP = 8        # how many recent messages to keep after compaction
SUMMARY_FILE = Path(".ir_agent_summary.json")


class ContextManager:
    def __init__(self):
        self.summary: str | None = self._load_summary()

    def maybe_compact(self, conversation: list[dict]) -> None:
        """Compact conversation if it exceeds the threshold."""
        if len(conversation) < COMPACTION_THRESHOLD:
            return

        # Summarize the older messages
        old_messages = conversation[:-MESSAGES_TO_KEEP]
        text_to_summarize = self._messages_to_text(old_messages)

        new_summary_piece = f"[Compacted {len(old_messages)} messages]\n{text_to_summarize}"
        if self.summary:
            self.summary = self.summary + "\n\n" + new_summary_piece
        else:
            self.summary = new_summary_piece

        # Trim the conversation in-place
        del conversation[:-MESSAGES_TO_KEEP]
        self._save_summary()

    def get_summary_message(self) -> str | None:
        """Return a formatted summary to inject as a system message."""
        if not self.summary:
            return None
        return (
            "## Conversation History Summary (compacted)\n"
            "The following is a summary of earlier parts of this conversation:\n\n"
            + self.summary
        )

    def _messages_to_text(self, messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if content:
                lines.append(f"{role.upper()}: {content[:300]}")
        return "\n".join(lines)

    def _load_summary(self) -> str | None:
        if SUMMARY_FILE.exists():
            try:
                data = json.loads(SUMMARY_FILE.read_text())
                return data.get("summary")
            except Exception:
                pass
        return None

    def _save_summary(self):
        SUMMARY_FILE.write_text(json.dumps({"summary": self.summary}, indent=2))

    def clear(self):
        self.summary = None
        if SUMMARY_FILE.exists():
            SUMMARY_FILE.unlink()
