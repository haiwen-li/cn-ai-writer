import os
from data_models import PostWithContext
from typing import List, Optional
from xai_sdk import Client
from xai_sdk.chat import user, image
from xai_sdk.tools import web_search, x_search
import json
from config import XAI_API_KEY, GROK_RESEARCH_OUTPUT_FILE

x_client = Client(api_key=XAI_API_KEY, timeout=180)


def _check_grok_citations(research: list, citations: List[str]):
    return [d for d in research if d["url"] in citations]


def grok_research(
    post_with_context: PostWithContext,
    output_file: str = GROK_RESEARCH_OUTPUT_FILE,
    save_to_file: bool = True,
    target_url_count: Optional[int] = 7,
) -> dict:
    post_id = post_with_context.post.post_id
    post_with_context_str = post_with_context.post_w_context_description
    media_urls = post_with_context.media_urls
    prompt = f"""Investigate whether the X post below is misleading. Images or video previews
from the post are provided if they exist.

Step 1: Understand the post and its context
- Find the post by post id {post_id} on X. If not found, search the post's main
  text on X.
- Identify the post author. Also note any signals that the account is
  parody/satire (e.g. bio, handle, verification).
- Fetch the thread and top comments. Understand how others interpret the post
  (e.g. joke/satire, potentially misleading, expressing an opinion).
- Summarize the post context in 1-2 sentences without rewording the post itself.
  Include who the OP is and how others interpret the post, especially if comments
  suggest it's a joke or satire, or provide potential fact-check directions.
  For example: "The OP is Sen. X. Many comments say his claim about X is
  unverified." or "The OP is a parody account, and commenters are laughing and
  treating it as a joke."

Step 2: Search for evidence
1. Search both the web and X for factual sources that refute or confirm the
   post's claims. Use the post context to guide your search if it could provide
   potential fact-check directions.
2. Aim for {target_url_count} pieces of evidence / URLs if possible.
3. For each source, include the URL and a brief note describing how it verifies
   or challenges the post. Include the publication date of the source if
   available.
4. Cover outlets across the ideological spectrum (left, center, right).
   Overlapping reasoning is acceptable when it comes from different publishers.
5. Prioritize evidence that is relevant, solid, and up to date.

Target post (ID: {post_id}):
{post_with_context_str}

Your response should be returned as a JSON object with the following structure:
```
{{
   "post_context": "one/two-sentence summary of the post context",
   "research": [
       {{"url": "url1", "description": "how the content of the URL fact-checks the post"}},
       ...
   ]
}}
```
If you cannot find sufficient evidence to fact-check the post, return an empty
research array."""
    message_content = [prompt]
    if media_urls:
        for media_url in media_urls:
            message_content.append(image(image_url=media_url, detail="high"))

    try:
        chat = x_client.chat.create(
            model="grok-4-fast",  # reasoning model
            tools=[
                web_search(enable_image_understanding=True),
                x_search(
                    enable_image_understanding=True, enable_video_understanding=True
                ),
            ],
        )
        chat.append(user(*message_content))
        response = chat.sample()
    except Exception as e:
        print(e)
        return None

    # is_thinking = True
    # print("Thinking...")
    # for response, chunk in chat.stream():
    #     # View the server-side tool calls as they are being made in real-time
    #     for tool_call in chunk.tool_calls:
    #         print(f"\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    #     if response.usage.reasoning_tokens and is_thinking:
    #         print(f"\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    #     if chunk.content and is_thinking:
    #         print("\n\nFinal Response:")
    #         is_thinking = False
    #     if chunk.content and not is_thinking:
    #         print(chunk.content, end="", flush=True)

    # print("\n\nCitations:")
    # print(response.citations)
    # print("\n\nUsage:")
    # print(response.usage)
    # print(response.server_side_tool_usage)
    # print("\n\nServer Side Tool Calls:")
    # print(response.tool_calls)

    # print("Final Response:")
    output = response.content
    citations = response.citations
    # print(output)

    try:
        research = json.loads(output)["research"]
        post_context = json.loads(output)["post_context"]
    except Exception as e:
        print(e)
        research = None
        post_context = None
    if research:
        valid_research = _check_grok_citations(research, citations)

        # Save valid research to JSONL file (when save_to_file is True)
        if valid_research:
            research_record = {
                "tweetId": post_with_context.post.post_id,
                "tweet_text": post_with_context_str,
                "post_context": post_context,
                "image_urls": media_urls if media_urls else [],
                "grok_research": valid_research,
            }

            if save_to_file and output_file:
                os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
                with open(output_file, "a") as f:
                    f.write(json.dumps(research_record) + "\n")

            return research_record
        else:
            return None
    else:
        return None
