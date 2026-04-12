import os
from datetime import datetime
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt


load_dotenv()


class SearchAPIError(Exception):
    """Recoverable search error."""


class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict, total=False):
    email_content: str
    sender_email: str
    email_id: str
    customer_id: str
    classification: EmailClassification | None
    search_results: list[str] | None
    customer_history: dict[str, Any] | None
    draft_response: str | None
    messages: list[Any] | None
    current_step: str | None
    sent_status: str | None


class MockEmailService:
    def __init__(self) -> None:
        self.sent_emails: list[dict[str, str]] = []

    def send(self, to_email: str, subject: str, body: str) -> None:
        self.sent_emails.append({"to": to_email, "subject": subject, "body": body})
        print(f"[EmailService] sent to={to_email} subject={subject}")


EMAIL_KB = {
    "password": [
        "Reset password via Settings > Security > Change Password.",
        "Password must be at least 12 characters with upper/lower/number/symbol.",
        "If 2FA is enabled, complete verification before password changes apply.",
    ],
    "billing": [
        "Refund window is 14 days for monthly plans unless regional law requires otherwise.",
        "Duplicate charges are typically reversed within 3-5 business days.",
        "Invoices can be downloaded from Billing > Invoice History.",
    ],
    "bug": [
        "Collect reproduction steps, expected behavior, and actual behavior.",
        "Include browser/app version and timestamp when reporting issues.",
        "Attach screenshots or logs when possible to speed investigation.",
    ],
    "feature": [
        "Feature requests are evaluated by impact, feasibility, and roadmap fit.",
        "Customers can track requests in Product Updates.",
    ],
}

MOCK_CUSTOMER_DB: dict[str, dict[str, Any]] = {
    "CUS-1001": {
        "tier": "pro",
        "subscription_status": "active",
        "last_invoice": "2026-03-30",
        "lifetime_value": 1190,
    },
    "CUS-2002": {
        "tier": "standard",
        "subscription_status": "active",
        "last_invoice": "2026-04-01",
        "lifetime_value": 199,
    },
}

email_service = MockEmailService()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "qwen3.5-27b"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


def _heuristic_classification(email_content: str) -> EmailClassification:
    text = email_content.lower()
    intent: Literal["question", "bug", "billing", "feature", "complex"] = "question"
    urgency: Literal["low", "medium", "high", "critical"] = "medium"

    if any(k in text for k in ["charged", "refund", "invoice", "billing", "payment"]):
        intent = "billing"
    elif any(k in text for k in ["error", "bug", "crash", "broken", "fail"]):
        intent = "bug"
    elif any(k in text for k in ["feature request", "please add", "enhancement"]):
        intent = "feature"
    elif len(email_content) > 900:
        intent = "complex"

    if any(k in text for k in ["urgent", "asap", "immediately", "critical"]):
        urgency = "high"
    if any(k in text for k in ["outage", "legal", "security breach", "fraud"]):
        urgency = "critical"

    return {
        "intent": intent,
        "urgency": urgency,
        "topic": "customer_support",
        "summary": email_content[:180].strip(),
    }


def _search_knowledge_base(query: str, intent: str) -> list[str]:
    if not query.strip():
        raise SearchAPIError("empty search query")
    key = intent if intent in EMAIL_KB else "password"
    docs = EMAIL_KB.get(key, [])
    if not docs:
        raise SearchAPIError("no matching documentation")
    return docs


def _create_bug_ticket(state: EmailAgentState) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"BUG-{ts}"


def _fetch_customer_history(customer_id: str) -> dict[str, Any]:
    return MOCK_CUSTOMER_DB.get(
        customer_id,
        {"tier": "standard", "subscription_status": "unknown", "last_invoice": "unknown"},
    )


def read_email(state: EmailAgentState) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=f"Processing email: {state.get('email_content', '')}")],
        "current_step": "email_read",
    }


def classify_intent(
    state: EmailAgentState,
) -> Command[Literal["search_documentation", "lookup_customer_history", "bug_tracking", "draft_response"]]:
    email_content = state.get("email_content", "")
    sender_email = state.get("sender_email", "")
    classification_prompt = f"""
Analyze this customer email and classify it.
Email: {email_content}
From: {sender_email}

Return:
- intent: one of [question, bug, billing, feature, complex]
- urgency: one of [low, medium, high, critical]
- topic: short topic
- summary: one-sentence summary
"""

    try:
        structured_llm = llm.with_structured_output(EmailClassification)
        classification = structured_llm.invoke(classification_prompt)
    except Exception:
        classification = _heuristic_classification(email_content)

    if classification["intent"] in ["question", "feature"]:
        goto = "search_documentation"
    elif classification["intent"] == "bug":
        goto = "bug_tracking"
    elif classification["intent"] == "billing":
        goto = "lookup_customer_history"
    else:
        goto = "draft_response"

    return Command(update={"classification": classification, "current_step": "classified"}, goto=goto)


def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    classification = state.get("classification") or _heuristic_classification(state.get("email_content", ""))
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}".strip()
    try:
        search_results = _search_knowledge_base(query=query, intent=classification.get("intent", "question"))
    except SearchAPIError as e:
        search_results = [f"Search temporarily unavailable: {e}"]

    return Command(
        update={"search_results": search_results, "current_step": "docs_retrieved"},
        goto="draft_response",
    )


def lookup_customer_history(state: EmailAgentState) -> Command[Literal["lookup_customer_history", "draft_response"]]:
    customer_id = state.get("customer_id")
    if not customer_id:
        user_input = interrupt(
            {
                "email_id": state.get("email_id", ""),
                "message": "Customer ID needed",
                "request": "Please provide customer_id for billing lookup.",
            }
        )
        return Command(
            update={"customer_id": user_input.get("customer_id", "")},
            goto="lookup_customer_history",
        )

    customer_data = _fetch_customer_history(customer_id)
    return Command(update={"customer_history": customer_data, "current_step": "customer_loaded"}, goto="draft_response")


def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    ticket_id = _create_bug_ticket(state)
    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created and queued for triage."],
            "current_step": "bug_tracked",
        },
        goto="draft_response",
    )


def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    classification = state.get("classification") or _heuristic_classification(state.get("email_content", ""))
    context_sections: list[str] = []

    if state.get("search_results"):
        formatted_docs = "\n".join([f"- {doc}" for doc in state["search_results"] or []])
        context_sections.append(f"Relevant knowledge:\n{formatted_docs}")

    if state.get("customer_history"):
        customer = state["customer_history"] or {}
        context_sections.append(
            f"Customer tier: {customer.get('tier', 'standard')}; subscription: {customer.get('subscription_status', 'unknown')}."
        )

    draft_prompt = f"""
Write a customer-support email reply.

Customer email:
{state.get('email_content', '')}

Intent: {classification.get('intent', 'unknown')}
Urgency: {classification.get('urgency', 'medium')}

{chr(10).join(context_sections)}

Requirements:
- Be concise, clear, and empathetic.
- If action is required, list concrete next steps.
- Do not invent policies not present in context.
"""

    try:
        response = llm.invoke(draft_prompt)
        draft = response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        draft = (
            "Thanks for reaching out. We received your request and are reviewing it now. "
            "Our support team will follow up with specific next steps shortly."
        )

    needs_review = classification.get("urgency") in ["high", "critical"] or classification.get("intent") in [
        "complex",
        "billing",
    ]
    goto = "human_review" if needs_review else "send_reply"
    return Command(update={"draft_response": draft, "current_step": "drafted"}, goto=goto)


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    classification = state.get("classification", {})
    human_decision = interrupt(
        {
            "email_id": state.get("email_id", ""),
            "original_email": state.get("email_content", ""),
            "draft_response": state.get("draft_response", ""),
            "urgency": classification.get("urgency"),
            "intent": classification.get("intent"),
            "action": "Please approve or edit before sending.",
        }
    )

    if human_decision.get("approved"):
        return Command(
            update={
                "draft_response": human_decision.get("edited_response", state.get("draft_response", "")),
                "current_step": "approved_by_human",
            },
            goto="send_reply",
        )
    return Command(update={"current_step": "handled_by_human"}, goto=END)


def send_reply(state: EmailAgentState) -> dict[str, str]:
    draft = state.get("draft_response", "")
    to_email = state.get("sender_email", "")
    email_service.send(to_email=to_email, subject="Re: Your support request", body=draft)
    return {"sent_status": "sent", "current_step": "replied"}


workflow = StateGraph(EmailAgentState)
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
)
workflow.add_node("lookup_customer_history", lookup_customer_history)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    initial_state: EmailAgentState = {
        "email_content": "I was charged twice for my subscription. This is urgent!",
        "sender_email": "customer@example.com",
        "email_id": "email_123",
        "messages": [],
    }
    config = {"configurable": {"thread_id": "customer_123"}}
    result = app.invoke(initial_state, config)
    print(f"Graph paused at interrupt: {result.get('__interrupt__')}")

    human_response = Command(
        resume={
            "customer_id": "CUS-1001",
            "approved": True,
            "edited_response": "Apologies for the duplicate charge. We have started a refund and it should complete within 3-5 business days.",
        }
    )
    final_result = app.invoke(human_response, config)
    print(f"Final state step: {final_result.get('current_step')}, send status: {final_result.get('sent_status')}")
