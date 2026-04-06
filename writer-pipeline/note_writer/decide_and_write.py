import os
from data_models import NoteResult, PostWithContext, ProposedMisleadingNote
from typing import List
import json
from openai import OpenAI
from config import OPENAI_API_KEY, PIPELINE_LOG_FILE
from note_writer.url_evaluator import check_all_urls_for_note
from note_writer.writer_util import (
    get_post_with_context_info_for_prompt,
    check_for_unsupported_media_in_post_with_context,
    get_misleading_tags,
    get_claimOpinion_score,
    clean_note,
)
from note_writer.grok_research import grok_research

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def decide_if_note_needed(
    post_with_context: PostWithContext, post_context: str, research_evidence: list[dict]
) -> str:
    """
    Decide whether a post needs a Community Note based on the provided evidence.

    Returns:
        - "WRITE NOTE" if a note should be written
        - "NO NOTE NEEDED" if the post doesn't need a note
        - "NOT ENOUGH EVIDENCE" if there's insufficient evidence
        - "SKIP" if the post contains video and the fact-check fundamentally depends on verifying the origin, identity, or authenticity of the video
    """
    post_with_context_str = post_with_context.post_w_context_description
    evidence_str = "\n".join(
        [f"{d['url']}: {d['description']}" for d in research_evidence]
    )
    system_prompt = """Decide whether a post needs a Community Note based on the provided evidence.

Returns:
    - "WRITE NOTE" if a note should be written
    - "NO NOTE NEEDED" if the post doesn't need a note
    - "NOT ENOUGH EVIDENCE" if there's insufficient evidence
"""

    user_prompt = f"""You are a Community Notes writer. Your job is to decide if the target post
could be perceived as misleading and whether it needs a community note to address
its issues. The output should be one of: "WRITE NOTE", "NO NOTE NEEDED", or
"NOT ENOUGH EVIDENCE".

Task rules
- Focus on the main claims of the post, not trivial errors.
- If media is included, use any legible text, recognizable logos/landmarks, and
  clearly identifiable public figures as part of the claim.
- If the target post is a quote/reply, evaluate the target post, using the
  quoted/replied post only as context.
- Use the context information (author info and audience reactions) to inform your
  decision. They are helpful for understanding whether the post is likely to be
  perceived as misleading.
- If you are unsure, err on the side of NO NOTE NEEDED.

Decision logic
Output NO NOTE NEEDED if:
- The OP is a satire/parody account, or the post is joking/sarcastic/exaggerated
  to be ironic, and commenters are interpreting it that way (with no strong signs
  of misunderstanding); or
- The post is mostly opinion, subjective takes, or personal experience; or
- The post contains no major factual claims, or the claims are not verifiable; or
- The post contains factual claims, but the provided evidence indicates those
  claims are accurate or not meaningfully misleading.

If the post contains major factual, verifiable claims and the evidence is
relevant:
- Output NOT ENOUGH EVIDENCE if the evidence is weak, mixed, or insufficient to
  confidently verify or refute the main claims. When unsure about the
  strength/sufficiency of the evidence, err on NOT ENOUGH EVIDENCE.
- Output WRITE NOTE only if the evidence clearly shows that the post's main
  claims are false or misleading in a way that could misinform a reasonable
  reader.

Post:
{post_with_context_str}

Additional context information about the post:
{post_context}

Evidence:
{evidence_str}

Output only one of: "WRITE NOTE", "NO NOTE NEEDED", or "NOT ENOUGH EVIDENCE".
Err on the side of NO NOTE NEEDED if unsure."""
    user_content = [{"type": "input_text", "text": user_prompt}]
    media_urls = post_with_context.media_urls
    if media_urls:
        for media_url in media_urls:
            user_content.append({"type": "image_url", "image_url": {"url": media_url}})
    try:
        response = openai_client.responses.create(
            model="gpt-5-mini-2025-08-07",
            instructions=system_prompt,
            input=[{"role": "user", "content": user_content}],
        )
        decision = response.output_text
        print(f"Decision: {decision}")
    except Exception as e:
        raise Exception(f"Error deciding if note needed: {e}")

    return decision


def write_note_with_research(
    post_with_context: PostWithContext,
    post_context: str,
    research_evidence: List[dict],
) -> str:
    """
    Write a note based on the provided evidence.
    """
    post_with_context_str = post_with_context.post_w_context_description
    evidence_str = "\n".join(
        [f"{d['url']}: {d['description']}" for d in research_evidence]
    )

    system_prompt = """You are a helpful fact-checking assistant.
Your goal is to write good Community Notes that would be approved helpful by
people with different viewpoints.
Do not invent facts or make claims that are not supported by the provided
evidence."""

    user_prompt = f"""Task: Write a community note for the target post below. Images or video
previews from the target post may be provided; if present, analyze any legible
text (OCR), recognizable logos/landmarks, and confidently identifiable public
figures to inform the note. Additional context provides post author details and
audience reactions. If the target post quotes/replies to another post, use it
only for context and focus on the target post.

Hard Constraints:
1. The note is written to explain why the post is misleading and add additional
   context to the post. Focus on primary claim(s) of the post rather than
   trivial details.
2. The note must be grounded in the provided evidence and should cite the URL of
   the evidence it uses. At least one URL must be cited.
3. Keep the note strictly under 280 characters. Stay neutral and clear.
4. No hashtags, emojis, unnecessary words. No markdown, brackets, or parentheses
   around URLs. Do not mention "this note" or "the prompt."

Target post:
```
{post_with_context_str}
```

Additional context about the post:
```
{post_context}
```

Allowed evidence sources:
```
{evidence_str}
```

Output only the final note (at most 280 characters).""".strip()

    user_content = [{"type": "input_text", "text": user_prompt}]
    media_urls = post_with_context.media_urls
    if media_urls:
        for image_url in media_urls:
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    try:
        response = openai_client.responses.create(
            model="gpt-5-mini-2025-08-07",
            instructions=system_prompt,
            input=[{"role": "user", "content": user_content}],
        )
        note_str = response.output_text
    except Exception as e:
        raise Exception(f"Error writing note: {e}")

    return note_str


def log_post_status(post_status: dict):
    if not PIPELINE_LOG_FILE:
        return
    os.makedirs(os.path.dirname(PIPELINE_LOG_FILE) or ".", exist_ok=True)
    with open(PIPELINE_LOG_FILE, "a") as f:
        f.write(json.dumps(post_status) + "\n")


def note_writing_pipeline(
    post_with_context: PostWithContext,
    account_name: str,
) -> NoteResult:
    post_id = post_with_context.post.post_id

    # (1) Check for unsupported media
    if check_for_unsupported_media_in_post_with_context(post_with_context):
        return NoteResult(
            post=post_with_context,
            error="Unsupported media type (e.g. video) found in post or in referenced post.",
        )

    # (2) Prepare post
    post_with_context = get_post_with_context_info_for_prompt(post_with_context)
    post_with_context_description = post_with_context.post_w_context_description

    # (3) Do grok research
    print("Conducting grok research...")
    research_record = grok_research(post_with_context, target_url_count=7)

    if research_record is None:
        log_post_status({"post_id": post_id, "status": "no_research_found"})
        return NoteResult(
            post=post_with_context,
            refusal="No valid research found.",
            context_description=post_with_context_description,
        )
    research_evidence = research_record.get("grok_research", [])
    if research_evidence is None or len(research_evidence) == 0:
        log_post_status({"post_id": post_id, "status": "no_research_found"})
        return NoteResult(
            post=post_with_context,
            refusal="No valid research found.",
            context_description=post_with_context_description,
        )
    post_context = research_record.get("post_context", "")
    print(f"Found {len(research_evidence)} pieces of evidence")

    # (4) Decide if note is needed and generate candidates
    print("Deciding if note is needed and generating candidates...")
    decision = decide_if_note_needed(post_with_context, post_context, research_evidence)

    if decision != "WRITE NOTE":
        log_post_status({"post_id": post_id, "status": decision})
        return NoteResult(
            post=post_with_context,
            refusal=decision,
            context_description=post_with_context_description,
        )

    # (5) Write note
    print("Writing note...")
    note_str = write_note_with_research(
        post_with_context=post_with_context,
        post_context=post_context,
        research_evidence=research_evidence,
    )

    note_str = clean_note(note_str)

    # (6) Score the note with X API
    claim_opinion_score = get_claimOpinion_score(account_name, note_str, post_id)
    print(f"Initial claim_opinion_score: {claim_opinion_score}")

    # (7) Apply scoring logic
    if claim_opinion_score < 0:
        # Score is too low - abandon
        log_post_status(
            {
                "post_id": post_id,
                "status": f"low CO score: {claim_opinion_score}",
                "note": note_str,
            }
        )
        return NoteResult(
            post=post_with_context,
            refusal=f"Claim opinion score is too low: {claim_opinion_score}.\nNote: {note_str}",
            context_description=post_with_context_description,
            claim_opinion_score=claim_opinion_score,
        )

    print(f"Note passed claim opinion score check: {claim_opinion_score}")

    # (8) Validate URLs
    print("Validating URLs in note...")
    evidence_urls = [d["url"] for d in research_evidence]
    if not check_all_urls_for_note(note_str, evidence_urls, check_with_requests=False):
        log_post_status(
            {
                "post_id": post_id,
                "status": "invalid urls",
                "note": note_str,
            }
        )
        return NoteResult(
            post=post_with_context,
            refusal=f"Some URLs in note are invalid.\nNote: {note_str}",
            context_description=post_with_context_description,
            claim_opinion_score=claim_opinion_score,
        )

    # (9) Get misleading tags
    print("Getting misleading tags...")
    misleading_tags = get_misleading_tags(post_with_context_description, note_str)

    # (10) Return successful result
    log_post_status(
        {
            "post_id": post_id,
            "status": "success",
            "note": note_str,
        }
    )
    return NoteResult(
        post=post_with_context,
        note=ProposedMisleadingNote(
            post_id=post_id,
            note_text=note_str,
            misleading_tags=misleading_tags,
        ),
        context_description=post_with_context_description,
        claim_opinion_score=claim_opinion_score,
    )
