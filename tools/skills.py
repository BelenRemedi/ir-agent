"""
Skills Tool - reads reusable skill/instruction files.

Skills are markdown files in ./skills/ that contain instructions
for how to perform specific tasks. The agent can list and read them.

This is inspired by OpenClaw's skills system.
"""

import os
from pathlib import Path


SKILLS_DIR = Path("skills")


class SkillsTool:
    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": (
                        "List all available skill files. Skills are reusable instructions "
                        "for specific tasks. Read a skill before performing that type of task."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_skill",
                    "description": "Read a specific skill file to get instructions for a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The skill filename (e.g. 'summarize.md')",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
        ]

    def list_skills(self) -> dict:
        if not SKILLS_DIR.exists():
            return {"skills": [], "message": "No skills directory found. Create ./skills/*.md files."}

        skills = []
        for f in sorted(SKILLS_DIR.glob("*.md")):
            # Read first line as description
            try:
                first_line = f.read_text().strip().split("\n")[0].lstrip("#").strip()
            except Exception:
                first_line = ""
            skills.append({"name": f.name, "description": first_line})

        return {"skills": skills, "count": len(skills)}

    def read_skill(self, name: str) -> dict:
        # Security: only allow filenames, no path traversal
        safe_name = Path(name).name
        skill_path = SKILLS_DIR / safe_name

        if not skill_path.exists():
            return {"error": f"Skill '{safe_name}' not found. Use list_skills to see available skills."}

        try:
            content = skill_path.read_text()
            return {"name": safe_name, "content": content}
        except Exception as e:
            return {"error": str(e)}
