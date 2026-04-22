from __future__ import annotations

import json
import re
from pathlib import Path

from harness.agents.secretary.wake import FalcoOrchestrator
from harness.config.config import FalcoSettings


def _safe_thread_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (raw or "").strip())
    cleaned = cleaned.strip("-_")
    return cleaned or "default"


def _list_threads(memory_root: Path) -> list[dict[str, str]]:
    if not memory_root.exists():
        return []
    items: list[dict[str, str]] = []
    for path in sorted(memory_root.glob("*.json")):
        if path.name == "evergreen.json":
            continue
        thread_id = path.stem
        updated_at = ""
        turns = 0
        summary = ""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            updated_at = str(raw.get("updated_at", "") or "")
            turns = len(raw.get("turns", []) or [])
            summary = str(raw.get("global_summary", "") or "").strip().replace("\n", " ")
            if len(summary) > 60:
                summary = summary[:57] + "..."
        except Exception:
            pass
        items.append(
            {
                "thread_id": thread_id,
                "updated_at": updated_at,
                "turns": str(turns),
                "summary": summary,
            }
        )
    items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return items


def _print_threads(threads: list[dict[str, str]]) -> None:
    if not threads:
        print("No existing sessions.")
        return
    print("Existing sessions:")
    for idx, item in enumerate(threads, start=1):
        info = f"{idx}. {item['thread_id']} (turns={item['turns']}"
        if item.get("updated_at"):
            info += f", updated={item['updated_at']}"
        info += ")"
        if item.get("summary"):
            info += f" - {item['summary']}"
        print(info)


def _choose_thread_id(settings: FalcoSettings) -> str:
    threads = _list_threads(settings.memory_root)
    _print_threads(threads)
    print("Select a session by number, or enter a new session name.")
    while True:
        raw = input("Session: ").strip()
        if not raw:
            if threads:
                print(f"Using most recent session: {threads[0]['thread_id']}")
                return threads[0]["thread_id"]
            print("Created new session: default")
            return "default"
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(threads):
                chosen = threads[idx - 1]["thread_id"]
                print(f"Using session: {chosen}")
                return chosen
            print("Invalid number, please choose from the list.")
            continue
        chosen = _safe_thread_id(raw)
        print(f"Created new session: {chosen}")
        return chosen


def main() -> None:
    settings = FalcoSettings.from_env()
    orchestrator = FalcoOrchestrator(settings)
    thread_id = _choose_thread_id(settings)
    print("Commands: /thread <name>, /sessions, quit")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if user_input == "/sessions":
            _print_threads(_list_threads(settings.memory_root))
            print(f"Current session: {thread_id}")
            continue
        if user_input.startswith("/thread "):
            thread_id = _safe_thread_id(user_input.replace("/thread ", "", 1).strip() or "default")
            print(f"Switched thread_id to: {thread_id}")
            continue
        output = orchestrator.invoke(user_input=user_input, thread_id=thread_id)
        print(f"Falco: {output}")


if __name__ == "__main__":
    main()
