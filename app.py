"""
Web UI for IR Agent using Flask.

Provides a chat interface at http://localhost:5000
"""

import os
import sys
import json
from flask import Flask, render_template_string, request, jsonify, session
from agent.agent import IRAgent
from tools.memory import MemoryTool
from tools.document_store import DocumentStoreTool

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global agent instance
_agent: IRAgent | None = None

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IR Agent</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg: #0d0f14;
    --surface: #161920;
    --border: #252933;
    --accent: #7c6af7;
    --accent2: #3ecfcf;
    --text: #e0e4f0;
    --muted: #6b7280;
    --user-bg: #1e2035;
    --agent-bg: #141822;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface);
  }

  header h1 {
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent2);
    border: 1px solid var(--accent2);
    padding: 2px 8px;
    border-radius: 20px;
    opacity: 0.7;
  }

  .sidebar {
    width: 260px;
    border-right: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .sidebar h2 {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    padding: 16px 16px 8px;
    font-family: 'JetBrains Mono', monospace;
  }

  .memory-list {
    flex: 1;
    overflow-y: auto;
    padding: 0 8px 8px;
  }

  .memory-item {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 4px;
    border: 1px solid var(--border);
    background: var(--bg);
  }

  .memory-key {
    color: var(--accent2);
    font-weight: 600;
    margin-bottom: 2px;
  }

  .memory-val { color: var(--muted); font-size: 0.7rem; }

  .main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .msg {
    max-width: 80%;
    animation: fadeUp 0.2s ease;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .msg.user { align-self: flex-end; }
  .msg.agent { align-self: flex-start; }

  .msg-bubble {
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 0.9rem;
    line-height: 1.6;
    white-space: pre-wrap;
  }

  .user .msg-bubble {
    background: var(--user-bg);
    border: 1px solid var(--accent);
    border-bottom-right-radius: 2px;
  }

  .agent .msg-bubble {
    background: var(--agent-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 2px;
  }

  .msg-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  .user .msg-label { text-align: right; }

  .thinking {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    background: var(--agent-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    border-bottom-left-radius: 2px;
    width: fit-content;
  }

  .dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: bounce 1s infinite;
  }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-6px); opacity: 1; }
  }

  .input-row {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
    background: var(--surface);
  }

  #user-input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 0.9rem;
    font-family: 'Syne', sans-serif;
    outline: none;
    transition: border-color 0.2s;
    resize: none;
    height: 46px;
  }

  #user-input:focus { border-color: var(--accent); }

  button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 0 20px;
    border-radius: 10px;
    cursor: pointer;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    transition: opacity 0.15s;
  }

  button:hover { opacity: 0.85; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }

  .docs-panel {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    padding: 8px 16px;
    border-top: 1px solid var(--border);
  }

  .doc-item { color: var(--accent2); }
</style>
</head>
<body>
<header>
  <h1>IR Agent</h1>
  <span class="badge">BM25 + Memory + Web Search</span>
</header>

<div class="main">
  <div class="sidebar">
    <h2>Working Memory</h2>
    <div class="memory-list" id="memory-list">
      <div style="padding:8px 10px;color:var(--muted);font-size:0.75rem;font-family:monospace">No memories yet</div>
    </div>
    <h2 style="border-top:1px solid var(--border);padding-top:12px">Documents</h2>
    <div class="memory-list" id="doc-list">
      <div style="padding:8px 10px;color:var(--muted);font-size:0.75rem;font-family:monospace">No documents indexed</div>
    </div>
  </div>

  <div class="chat-area">
    <div id="messages">
      <div class="msg agent">
        <div class="msg-label">IR Agent</div>
        <div class="msg-bubble">Hello! I'm an AI agent with Information Retrieval capabilities.

I can:
• 🔍 Search the web for current information
• 📝 Remember things across our conversations (persistent memory)
• 📚 Index and search documents using BM25 ranking
• 🐍 Execute Python for calculations
• 📖 Use skill files for specialized tasks

What would you like to do?</div>
      </div>
    </div>

    <div class="input-row">
      <textarea id="user-input" placeholder="Ask anything..." rows="1"></textarea>
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>
</div>

<script>
const input = document.getElementById('user-input');
const msgs = document.getElementById('messages');

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  document.getElementById('send-btn').disabled = true;

  appendMsg('user', text);
  const thinking = appendThinking();
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    thinking.remove();
    appendMsg('agent', data.response || data.error || 'Error');
    await refreshSidebar();
  } catch (e) {
    thinking.remove();
    appendMsg('agent', 'Error: ' + e.message);
  }

  document.getElementById('send-btn').disabled = false;
  msgs.scrollTop = msgs.scrollHeight;
}

function appendMsg(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="msg-label">${role === 'user' ? 'You' : 'IR Agent'}</div>
    <div class="msg-bubble">${escHtml(text)}</div>`;
  msgs.appendChild(div);
  return div;
}

function appendThinking() {
  const div = document.createElement('div');
  div.className = 'msg agent';
  div.innerHTML = `<div class="msg-label">IR Agent</div>
    <div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  msgs.appendChild(div);
  return div;
}

function escHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/"/g,'&quot;').replace(/\\n/g,'<br>');
}

async function refreshSidebar() {
  try {
    const r = await fetch('/sidebar');
    const d = await r.json();

    const memEl = document.getElementById('memory-list');
    if (d.memory.length === 0) {
      memEl.innerHTML = '<div style="padding:8px 10px;color:var(--muted);font-size:0.75rem;font-family:monospace">No memories yet</div>';
    } else {
      memEl.innerHTML = d.memory.map(m =>
        `<div class="memory-item"><div class="memory-key">${escHtml(m.key)}</div><div class="memory-val">${escHtml(m.value.substring(0,80))}${m.value.length > 80 ? '…' : ''}</div></div>`
      ).join('');
    }

    const docEl = document.getElementById('doc-list');
    if (d.documents.length === 0) {
      docEl.innerHTML = '<div style="padding:8px 10px;color:var(--muted);font-size:0.75rem;font-family:monospace">No documents indexed</div>';
    } else {
      docEl.innerHTML = d.documents.map(doc =>
        `<div class="memory-item"><div class="memory-key doc-item">${escHtml(doc.title)}</div><div class="memory-val">${doc.chunks} chunks</div></div>`
      ).join('');
    }
  } catch(e) {}
}

refreshSidebar();
</script>
</body>
</html>
"""


def get_agent():
    global _agent
    if _agent is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BERGET_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY or BERGET_API_KEY environment variable")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        _agent = IRAgent(api_key=api_key, base_url=base_url, model=model)
    return _agent


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"})
    try:
        agent = get_agent()
        response = agent.chat(message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sidebar")
def sidebar():
    mem_tool = MemoryTool()
    mem_data = mem_tool.memory_read()

    doc_tool = DocumentStoreTool()
    doc_data = doc_tool.document_list()

    return jsonify({
        "memory": mem_data["memory"],
        "documents": doc_data["documents"],
    })


@app.route("/reset", methods=["POST"])
def reset():
    get_agent().reset()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Web UI: http://localhost:{port}")
    app.run(debug=False, port=port)
