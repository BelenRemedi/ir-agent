# IR Agent

An AI agent extended with Information Retrieval capabilities, built for the Information Retrieval course — Assignment 2.

## Overview

IR Agent wraps a large language model in an agentic loop and gives it tools for finding, storing, and retrieving context. The core hypothesis is that **an LLM with good IR tools is far more useful than one without**: it can remember things, search documents efficiently, and fetch live information from the web.

```
┌─────────────────────────────────────────────────────────────┐
│                        IR Agent Loop                        │
│                                                             │
│  User Input → [Build Context] → LLM → Tool Call? → Result  │
│                    ↑                                   │     │
│                    └───────────────────────────────────┘     │
│                                                             │
│  Context Sources:                                           │
│  • Compacted conversation summary (long-term context)       │
│  • Working memory (persistent key-value notes)              │
│  • BM25 document index (local knowledge base)               │
│  • Web search results (live information)                    │
│  • Skill files (reusable task instructions)                 │
└─────────────────────────────────────────────────────────────┘
```

## Features

### IR Methods Implemented

| Feature | Description | IR Concept |
|---------|-------------|------------|
| **BM25 Document Search** | Indexes documents in chunks, ranks by BM25 score | Classical probabilistic IR |
| **Working Memory** | Persistent key-value store with keyword search | Information management |
| **Web Search** | DuckDuckGo fallback, Brave Search with API key | Web IR |
| **Conversation Compaction** | Summarizes old messages to preserve context | Context window management |
| **Skill Files** | Markdown files with task instructions the agent reads on-demand | Knowledge retrieval |

### BM25 Implementation

BM25 (Best Match 25) is implemented from scratch in `tools/document_store.py`. Given a query Q and document D:

```
score(D, Q) = Σ IDF(t) × tf(t,D) × (k1+1) / (tf(t,D) + k1×(1 - b + b×|D|/avgdl))

where:
  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  k1 = 1.5  (term frequency saturation)
  b  = 0.75 (document length normalization)
```

Documents are chunked (400 chars, 80-char overlap) before indexing to enable passage-level retrieval.

### Agent Architecture

```
main.py / ui/app.py
      │
      ▼
agent/agent.py          ← Core agentic loop (LLM + tool dispatch)
      │
      ├── agent/context.py     ← Conversation compaction + summary injection
      │
      └── tools/
            ├── registry.py        ← Tool registration + dispatch
            ├── memory.py          ← Persistent working memory
            ├── document_store.py  ← BM25 index + chunking
            ├── web_search.py      ← DuckDuckGo / Brave Search
            ├── python_exec.py     ← Safe Python execution
            └── skills.py          ← Skill file reader
```

The agent follows a **ReAct**-style loop:
1. Receive user message
2. Inject system prompt + conversation summary + recent history
3. Call LLM with tool schemas
4. If LLM returns tool calls → execute → append results → repeat
5. If LLM returns text → return to user

## Quickstart

### Prerequisites
- Python 3.11+
- An API key (OpenAI format) — **Berget.AI** gives free credits with code `HACKTHON26`

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ir-agent.git
cd ir-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API key and model
```

### Configure `.env`

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.berget.ai/v1     # or https://api.openai.com/v1
OPENAI_MODEL=llama-3.3-70b-instruct          # or gpt-4o-mini
```

> **Berget.AI setup**: Sign up at [berget.ai](https://berget.ai), use code `HACKTHON26` for free credits. Get your API key from the dashboard. The base URL is `https://api.berget.ai/v1`.

### Run CLI

```bash
source .env  # or: export $(cat .env | xargs)

# Interactive chat
python main.py

# Single query
python main.py --query "What is BM25?"

# Demo showcasing all IR tools
python main.py --demo
```

### Run Web UI

```bash
source .env
python ui/app.py
# Open http://localhost:5000
```

The web UI shows a live sidebar with:
- **Working Memory** — all saved key-value notes
- **Documents** — indexed document titles and chunk counts

## Usage Examples

### Web Search
```
You: What are the latest developments in vector search databases?
Agent: [uses web_search] ...
```

### Index and Search Documents
```
You: Please index this: "Retrieval-Augmented Generation (RAG) combines a retrieval 
     component with a generative model. The retriever finds relevant documents 
     from a corpus using dense or sparse methods..."
     Title: "RAG Overview"

You: What does RAG stand for and how does it work?
Agent: [uses document_search → finds indexed chunk] ...
```

### Persistent Memory
```
Session 1:
You: Remember that my research focus is on neural IR models
Agent: [uses memory_write] Saved!

Session 2 (later):
You: What am I researching?
Agent: [uses memory_read] Your research focus is on neural IR models.
```

### Python Calculations
```
You: Calculate the IDF for a term appearing in 3 of 1000 documents
Agent: [uses python_exec]
      IDF = log((1000 - 3 + 0.5) / (3 + 0.5) + 1) ≈ 5.71
```

### Skills
```
You: Use the research skill to investigate TF-IDF vs BM25
Agent: [reads research.md skill → follows structured research process]
```

## Extending the Agent

### Add a New Tool

1. Create `tools/my_tool.py`:
```python
class MyTool:
    def schemas(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "my_tool_name",
                "description": "What this tool does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "..."}
                    },
                    "required": ["input"]
                }
            }
        }]
    
    def my_tool_name(self, input: str) -> dict:
        # Your logic here
        return {"result": ...}
```

2. Register in `tools/registry.py`:
```python
from tools.my_tool import MyTool
# Add MyTool() to the all_tools list
```

### Add a New Skill

Create a markdown file in `skills/`:
```markdown
# My Skill Name

Use this skill when [situation].

## Instructions
1. Step one
2. Step two
...
```

The agent will see it via `list_skills` and read it via `read_skill`.

### Better Web Search

Set `BRAVE_SEARCH_API_KEY` in your `.env`. Get a free key at [brave.com/search/api](https://brave.com/search/api/).

## Architecture Decisions

### Why BM25 over TF-IDF?
BM25 adds two improvements over TF-IDF:
- **Term saturation**: repeated terms give diminishing returns (controlled by k1)
- **Length normalization**: shorter documents aren't penalized vs longer ones (controlled by b)

### Why chunking?
Long documents are split into overlapping chunks so retrieval returns the *specific passage* relevant to a query rather than the whole document.

### Why compaction?
LLMs have finite context windows. Compaction summarizes old conversation turns and keeps only the recent ones, allowing arbitrarily long conversations without losing context.

### Why skill files?
Skills decouple *how to do tasks* from the agent's system prompt. You can update or add skills without changing any code.

## Project Structure

```
ir-agent/
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies (openai, flask only)
├── .env.example         # Configuration template
├── .gitignore
├── agent/
│   ├── agent.py         # Core agentic loop
│   └── context.py       # Compaction + context management
├── tools/
│   ├── registry.py      # Tool dispatch
│   ├── memory.py        # Persistent memory
│   ├── document_store.py # BM25 IR engine
│   ├── web_search.py    # Web search
│   ├── python_exec.py   # Code execution
│   └── skills.py        # Skill file reader
├── ui/
│   └── app.py           # Flask web interface
└── skills/
    ├── summarize.md      # Summarization skill
    └── research.md       # Research skill
```

## Dependencies

The project intentionally has minimal dependencies:
- `openai` — OpenAI-compatible API client (works with Berget.AI, Ollama, etc.)
- `flask` — Web UI (optional; only needed for `ui/app.py`)

BM25 is implemented from scratch with no external IR libraries.

## License

MIT
