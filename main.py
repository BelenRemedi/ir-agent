#!/usr/bin/env python3
"""
IR Agent - CLI entry point.

Usage:
    python main.py                          # interactive chat
    python main.py --query "What is BM25?" # single query
    python main.py --demo                  # run a demo
"""

import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()


def get_config() -> tuple[str, str, str]:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BERGET_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY or BERGET_API_KEY environment variable.")
        sys.exit(1)

    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",  # change to Berget URL if needed
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return api_key, base_url, model


def run_interactive(agent):
    """Interactive REPL loop."""
    print("\n🤖 IR Agent — type 'exit' to quit, 'reset' to clear conversation, 'memory' to view memory\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[Conversation reset]\n")
            continue
        if user_input.lower() == "memory":
            from tools.memory import MemoryTool
            mem = MemoryTool()
            result = mem.memory_read()
            print(f"\n📝 Memory ({result['count']} entries):")
            for item in result['memory']:
                print(f"  [{item['key']}] {item['value'][:80]}")
            print()
            continue

        print("Agent: ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)
        print()


def run_demo(agent):
    """Run a scripted demo showcasing IR capabilities."""
    demo_steps = [
        (
            "web_search",
            "What is BM25 and how is it used in information retrieval?",
        ),
        (
            "document_index",
            (
                "Please index this text: "
                "'BM25 (Best Match 25) is a ranking function used by search engines. "
                "It improves on TF-IDF by normalizing document length. "
                "The k1 parameter controls term frequency saturation, "
                "while b controls document length normalization. "
                "BM25 is the default ranking in Elasticsearch and Lucene.'"
                " with title 'BM25 Overview'"
            ),
        ),
        (
            "document_index_2",
            (
                "Please index this text: "
                "'Automobiles use internal combustion engines to convert fuel into "
                "mechanical motion. The pistons compress a fuel-air mixture which "
                "ignites and drives the crankshaft, propelling the vehicle forward.' "
                "with title 'Vehicle Mechanics'"
            ),
        ),
        (
            "document_search",
            "Search my documents for information about term frequency normalization",
        ),
        (
            "query_expansion_demo",
            (
                "Search my documents for: how do cars work? "
                "Note: the document uses the word 'automobiles' not 'cars' — "
                "this tests whether query expansion finds it anyway."
            ),
        ),
        (
            "memory",
            "Remember that I am studying Information Retrieval at university, assignment 2",
        ),
        (
            "memory_search",
            "What do you know about my studies?",
        ),
        (
            "skills",
            "List your available skills",
        ),
        (
            "calculation",
            "Calculate: what is log((100 - 5 + 0.5) / (5 + 0.5) + 1) — this is the IDF for a term appearing in 5 out of 100 docs",
        ),
    ]

    print("\n🎬 Running IR Agent Demo\n" + "=" * 40)
    for step_name, query in demo_steps:
        print(f"\n[Demo: {step_name}]\nYou: {query}")
        print("Agent: ", end="", flush=True)
        response = agent.chat(query)
        print(response)
        print()


def main():
    parser = argparse.ArgumentParser(description="IR Agent - AI agent with Information Retrieval tools")
    parser.add_argument("--query", "-q", help="Single query mode")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--base-url", help="API base URL (overrides OPENAI_BASE_URL)")
    parser.add_argument("--model", help="Model name (overrides OPENAI_MODEL)")
    args = parser.parse_args()

    api_key, base_url, model = get_config()
    if args.base_url:
        base_url = args.base_url
    if args.model:
        model = args.model

    print(f"🔧 Using model: {model} @ {base_url}")

    # Import here so config errors show first
    from agent.agent import IRAgent
    agent = IRAgent(api_key=api_key, base_url=base_url, model=model)

    if args.query:
        response = agent.chat(args.query)
        print(response)
    elif args.demo:
        run_demo(agent)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
