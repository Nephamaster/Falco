from __future__ import annotations

from config import FalcoSettings
from orchestrator import FalcoOrchestrator


def main() -> None:
    orchestrator = FalcoOrchestrator(FalcoSettings.from_env())
    thread_id = "default"
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        if user_input.startswith("/thread "):
            thread_id = user_input.replace("/thread ", "", 1).strip() or "default"
            print(f"Switched thread_id to: {thread_id}")
            continue
        output = orchestrator.invoke(user_input=user_input, thread_id=thread_id)
        print(f"Falco: {output}")


if __name__ == "__main__":
    main()
