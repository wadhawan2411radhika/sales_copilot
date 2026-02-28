"""
llm/client.py
-------------
Thin LLM wrapper + all prompt templates in one place.

Design:
  - One function per query intent (summarise, qa, sentiment_filter)
  - System prompt is constant; user prompt is dynamically built
  - Every response request specifies max_tokens appropriate to the task
  - Citations are injected as part of the context, not hallucinated
"""

import os
from openai import OpenAI

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client


CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")   # cheap + fast default


# ── Shared system prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Sales Intelligence Copilot that helps sales teams 
understand their call recordings. You have access to structured sales call transcripts.

Your responses must:
1. Be grounded ONLY in the provided transcript context
2. Always cite sources using the format: [Call ID, Timestamp] e.g. [call_002, 01:24]
3. Be concise and structured — use bullet points for lists, prose for summaries
4. Clearly distinguish between vendor statements and prospect statements
5. If information is not in the context, say "Not found in available transcripts"

Never hallucinate details, prices, names, or commitments not in the context."""


# ── Intent: Summarise a call ───────────────────────────────────────────────────

def summarise_call(call_id: str, transcript_text: str) -> str:
    prompt = f"""Summarise this sales call ({call_id}).

Structure your summary as:
## Overview
(1-2 sentences: who, what stage, outcome)

## Key Discussion Points
(bullet list of main topics covered)

## Objections Raised
(prospect objections + how vendor responded)

## Commitments & Action Items
(what each party agreed to do, with timestamps)

## Deal Status
(where the deal stands after this call)

TRANSCRIPT:
{transcript_text}
"""
    return _chat(prompt, max_tokens=1000)


# ── Intent: Q&A with citations ─────────────────────────────────────────────────

def answer_with_citations(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[{c['call_id']}, {c['start_time']}–{c['end_time']}] "
            f"(topic: {c.get('topic_tag','?')}):\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Answer the following question using ONLY the transcript excerpts below.
For every claim, cite the source as [call_id, timestamp].

QUESTION: {query}

TRANSCRIPT EXCERPTS:
{context}

ANSWER:"""
    return _chat(prompt, max_tokens=600)


# ── Intent: Sentiment-filtered retrieval ───────────────────────────────────────

def filter_negative_sentiment(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[{c['call_id']}, {c['start_time']}–{c['end_time']}]:\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""From the transcript excerpts below, extract ONLY comments that express:
- Negative sentiment (concern, frustration, skepticism, rejection)
- Competitive pressure (comparisons to competitors that disadvantage us)
- Unresolved objections
- Budget or cost resistance

Topic focus: {query}

For each negative comment:
- Quote the relevant line(s)
- Cite [call_id, timestamp]  
- Label the sentiment type (e.g., "Price resistance", "Competitor anchor", "Feature gap concern")

TRANSCRIPT EXCERPTS:
{context}

NEGATIVE COMMENTS:"""
    return _chat(prompt, max_tokens=800)


# ── Intent: Ingestion confirmation ────────────────────────────────────────────

def summarise_ingestion(result: dict) -> str:
    if result["status"] == "skipped":
        return (f"⚠️  '{result['filename']}' was already ingested as {result['call_id']}. "
                f"Skipping duplicate.")
    if result["status"] == "error":
        return f"❌  Error ingesting '{result['filename']}': {result['reason']}"

    return (
        f"✅  Ingested '{result['filename']}' as **{result['call_id']}**\n"
        f"   • Type: {result['call_type']}\n"
        f"   • Utterances parsed: {result['utterances']}\n"
        f"   • Chunks created: {result['chunks']}\n"
        f"   • Chunks indexed: {result['chunks_indexed']}\n"
        f"   • Action items found: {result['action_items_found']}"
    )


# ── Core chat call ─────────────────────────────────────────────────────────────

def _chat(user_prompt: str, max_tokens: int = 600) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,   # low temp = more factual, less creative
    )
    return response.choices[0].message.content.strip()