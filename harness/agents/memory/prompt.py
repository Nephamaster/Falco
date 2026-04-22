SCORE_IMPORTANCE_PROMPT = """You are evaluating how important a dialogue turn is for long-term memory.

Dialogue:
{dialogue}

Evaluate its importance for future context continuity.

Return your answer in the following JSON format ONLY:
{
  "score": <integer from 1 to 10>,
  "reason": "<brief explanation>"
}

Scoring guidelines:
- 9-10: Critical information (user goals, long-term plans, key decisions, constraints, personal facts, ongoing tasks)
- 7-8: Important context (preferences, repeated behaviors, notable intermediate decisions)
- 4-6: Moderately useful (contextual but not essential information)
- 1-3: Low importance (casual conversation, transient details, generic Q&A)

Additional rules:
- Score must be an INTEGER between 1 and 10
- Be conservative: assign high scores when the information is likely to matter in future interactions
- Reason should be concise and specific (1 sentence)
- Do NOT include any text outside the JSON

Return ONLY valid JSON."""


GLOBAL_SUMMARY_PROMPT="""You maintain a compact, continuously updated summary for dialogue memory.
Your goal is to preserve only DURABLE and HIGH-IMPORTANCE information for future interactions.
Update the summary using the new dialogue turn.

**Current Summary**:
{summary}
**New Dialogue Turn**:
{dialogue}

Return your answer in the following JSON format ONLY:
{
  "summary": <updated summary>
}

Guidelines:
- Keep only important information: user goals, preferences, constraints, decisions, key facts, and open tasks
- Remove or ignore trivial, redundant, or short-term details
- Merge with existing summary instead of appending blindly (deduplicate and compress)
- If new information contradicts old information, **keep the latest and discard outdated content**
- Use the importance score to guide updates
    - >=7: likely should be included
    - 4-6: include only if it adds new value
    - <=3: usually ignore unless it connects to existing important context
- Prefer abstraction over verbatim copying
- Keep the summary concise and well-structured

Return ONLY valid JSON."""


SILENT_COMPRESS_PROMPT = """You are performing a silent context-maintenance step for a dialogue system.
The active context is near its limit. Some context will be removed regardless of importance.

Your goal is to preserve critical information BEFORE it is lost.

You must do THREE things:
1. Compress the **current summary** so that important context is retained in a smaller form
2. Decide whether any valuable information should be written to the **Daily Log**
3. Decide whether any durable user-profile information should be written to the **Evergreen Diary**

Key principle:
- Any part of the context (even important turns) may be dropped due to window limits
- You must proactively extract and preserve information that should survive this pruning step

Memory routing rules:
- Compressed Summary:
  Preserve information required for future continuity (goals, constraints, decisions, unresolved tasks, key context)

- Daily Log:
  Store recent important events, progress, decisions, temporary constraints, or ongoing tasks that may still matter later

- Evergreen Diary:
  Store only stable, long-term user traits (preferences, habits, interests, long-term goals, persistent constraints)

What to prioritize extracting:
- Important facts that appear ONLY in droppable context
- Information not yet captured in the summary
- Newly introduced constraints or decisions
- Ongoing tasks or partially completed work

What to avoid:
- Blind copying of context
- Redundant information already well represented in summary
- Small talk or stylistic filler
- Fully resolved, non-reusable details

Conflict handling:
- If new information conflicts with old summary, keep the latest version

Return ONLY valid JSON with exactly these fields:
{
  "compressed_summary": string,
  "write_daily": boolean,
  "daily_note": string,
  "write_evergreen": boolean,
  "evergreen_note": string
}

Output requirements:
- `compressed_summary` should be concise, information-dense, and written in clear prose
- `daily_note` and `evergreen_note` should each be short and self-contained
- If no write is needed, use `false` and an **empty string** for the corresponding note

Return ONLY valid JSON."""


SILENT_COMPRESS_PAYLOAD_TEMPLATE = """Current Summary:
{summary}

Latest Interaction:
- User: {latest_user}
- Assistant: {latest_assistant}

At-Risk Context (may be removed due to context window limits, even if important):
{critical_turns}

Full Context Snapshot (reference only; extract key information, DO NOT copy verbatim):
{context_snapshot}

Instruction:
- Assume the "At-Risk Context" may be permanently lost after this step
- Identify important information that is NOT yet preserved
- Decide whether to store it in summary, daily log, or evergreen diary
- Focus on preserving information, not wording
"""


REFLECTION_DECISION_PROMPT = """You are Falco's reflexion module. Extract one reusable operational lesson from the latest turn.
Write only if the lesson will improve future agent behavior, tool choice, validation, planning, or error recovery.
Do not store user private facts here; those belong to user memory.
Return JSON: should_write, lesson, trigger, recommendation, confidence, tags.
Keep lesson and recommendation concise."""


REFLECTION_DECISION_PAYLOAD_TEMPLATE = """User:
{user}

Assistant:
{assistant}

Tool observations:
{observations}"""


DAILY_LOG_DECISION_PROMPT = """You are deciding whether the current dialogue turn should be written into a structured Daily Log record.

The Daily Log is short-term episodic memory. It stores useful information that may matter in near-future interactions, especially if active context is later pruned.

Your job:
1. Decide whether this turn is worth writing
2. If yes, extract concise, reusable, structured information
3. Abstract and compress; do not copy the chat

Decision rules:
- Be conservative
- Write only if the turn contains information likely to be useful later
- Use the importance score as a strong signal:
  - importance >= 7: likely worth writing if any concrete useful content exists
  - importance 4-6: write only if non-trivial new information is present
  - importance <= 3: usually do not write

Suitable Daily Log content:
- important facts
- decisions
- tasks
- temporary constraints
- notable user preferences relevant to ongoing work
- mentioned artifacts
- next actions

Do not write:
- small talk
- generic explanation
- repeated context
- fully resolved low-value details
- content with no plausible future use

Return JSON ONLY with exactly these fields:
{
  "should_write": boolean,
  "summary": string,
  "category": string,
  "confidence": float,
  "facts": string[],
  "decisions": string[],
  "tasks": string[],
  "user_preferences": string[],
  "constraints": string[],
  "artifacts": string[],
  "next_actions": string[],
  "tags": string[]
}

Rules:
- If should_write is false:
  - summary = ""
  - all list fields = []
- If should_write is true:
  - at least one of summary, facts, decisions, tasks, user_preferences, constraints, artifacts, next_actions must be non-empty
- summary must be one concise sentence, <= 40 words
- category must be exactly one of:
  ["task", "decision", "progress", "constraint", "preference", "artifact", "info", "other"]
- confidence must be a float from 0.0 to 1.0
- Every list item must be concise, atomic, and non-redundant
- tags must be short keywords

Return ONLY valid JSON."""

DAILY_LOG_DECISION_PAYLOAD_TEMPLATE = """Importance Score: {importance}

Current Turn:
- User: {user}
- Assistant: {assistant}

Instruction:
Decide whether this turn should be stored in the Daily Log.
If yes, extract only reusable structured information.
If not, return an empty record following the schema."""


EVERGREEN_DECISION_PROMPT = """Decide whether this dialogue turn contains durable user-profile information worth writing to Evergreen Diary.

Evergreen is for long-term stable memory only:
- persistent preferences, habits, interests
- long-term goals
- stable constraints

Do NOT write:
- temporary requests
- transient task status
- generic chat
- repeated already-known facts

Return JSON ONLY:
{
  "should_write": boolean,
  "note": string,
  "confidence": float,
  "tags": string[]
}

Rules:
- If should_write=false: note="", tags=[]
- Keep note concise and atomic (<=30 words)
- confidence in [0,1]
- tags should be short keywords"""


EVERGREEN_DECISION_PAYLOAD_TEMPLATE = """Importance Score: {importance}

Current Turn:
- User: {user}
- Assistant: {assistant}"""
