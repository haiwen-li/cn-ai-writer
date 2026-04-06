import html
import re
from data_models import Post, PostWithContext, MisleadingTag
from typing import List
import json
from openai import OpenAI
from requests_oauthlib import OAuth1Session
from config import OPENAI_API_KEY, X_API_ACCOUNTS

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def unescape(text: str) -> str:
    """Remove layers of HTML escaping so the text matches natural language."""
    return html.unescape(html.unescape(text)) if isinstance(text, str) else text


def _check_for_unsupported_media(post: Post) -> bool:
    """Check if the post contains unsupported media types."""
    for media in post.media:
        if media.media_type not in ["photo"]:
            return True
    return False


def check_for_unsupported_media_in_post_with_context(
    post_with_context: PostWithContext,
) -> bool:
    """Check if any referenced posts contain unsupported media types. The target post is Okay to have video media."""
    # if _check_for_unsupported_media(post_with_context.post):
    #     return True
    if post_with_context.quoted_post and _check_for_unsupported_media(post_with_context.quoted_post):
        return True
    if post_with_context.in_reply_to_post and _check_for_unsupported_media(post_with_context.in_reply_to_post):
        return True
    return False


def _describe_image(image_url: str) -> str:
    prompt = """
    Describe the image. Please include:
    - A concise caption("A red-brick lighthouse on a rocky shoreline at dusk")
    - Any legible text (OCR)
    - Any recognizable logos, landmarks, and geographical features (e.g., skyline silhouettes) that help with landmark matching
    - Any faces/celebrities you can confidently identify.
    Stay neutral.
    
    Only respond with the description, no other text.
    """
    try:
        response = openai_client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )
        description = response.output_text
    except Exception as e:
        raise Exception(f"Error describing image: {e}")

    return description


def _summarize_images_for_post(post: Post) -> str:
    """
    Summarize images, if they exist. Abort if video or other unsupported media type.
    """
    images_summary = ""
    for i, media in enumerate(post.media):
        assert media.media_type == "photo"  # remove assert when video support is added
        image_description = _describe_image(media.url)
        images_summary += f"Image {i + 1}: {image_description}\n"
    return images_summary


def get_post_with_context_info_for_prompt(
    post_with_context: PostWithContext,
) -> PostWithContext:
    post_text = unescape(post_with_context.post.text)
    created_at = post_with_context.post.created_at.strftime("%Y-%m-%d")
    description = f"""The target post (posted on {created_at}):```\n{post_text}\n```"""

    # if images or videos are presented in the main post, directly add them as input as media_urls
    media_urls = []
    if post_with_context.post.media:
        for media in post_with_context.post.media:
            if media.media_type == "photo":
                media_urls.append(media.url)
            elif media.media_type == "video":
                media_urls.append(media.preview_image_url)
    post_with_context.media_urls = media_urls

    # images_summary = _summarize_images_for_post(post_with_context.post)
    # if images_summary is not None and len(images_summary) > 0:
    #     description += f"""Summary of images in the post:\n```\n{images_summary}\n``` """

    if post_with_context.quoted_post:
        quoted_post_text = unescape(post_with_context.quoted_post.text)
        description += f"""The post of interest had quoted (referenced) another post. Here is the quoted post's text:```\n{quoted_post_text}\n```"""

        quoted_images_summary = _summarize_images_for_post(
            post_with_context.quoted_post
        )
        if quoted_images_summary is not None and len(quoted_images_summary) > 0:
            description += f"""Summary of images in the quoted post:```\n{quoted_images_summary}\n```"""

    if post_with_context.in_reply_to_post:
        replied_to_post_text = unescape(post_with_context.in_reply_to_post.text)
        description += f"""The post of interest was a reply to another post. Here is the replied-to post's text:```\n{replied_to_post_text}\n```"""

        replied_to_images_summary = _summarize_images_for_post(
            post_with_context.in_reply_to_post
        )
        if replied_to_images_summary is not None and len(replied_to_images_summary) > 0:
            description += f"""Summary of images in the replied-to post:```\n{replied_to_images_summary}\n```"""

    post_with_context.post_w_context_description = description
    return post_with_context


def clean_note(text: str) -> str:
    """
    Removing markdown-style links ([text](url)) and anything unnecessary.
    """
    # Remove markdown-style links, keep the inner text (label)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\2", text)
    # Remove any leftover () around URLs
    text = re.sub(r"\((https?://[^\s)]+)\)", r"\1", text)
    # Replace oai url
    text = text.replace("?utm_source=openai", "")
    return text.strip()


def get_misleading_tags(
    post_with_context_description: str, note_text: str, retries: int = 3
) -> List[MisleadingTag]:
    misleading_why_tags_prompt = f"""Below will be a post on X, and a proposed community note that \
        adds additional context to the potentially misleading post. \
        Your task will be to identify which of the following tags apply to the post and note. \
        You may choose as many tags as apply, but you must choose at least one. \
        You must respond in valid JSON format, with a list of which of the following options apply:
        - "factual_error":  # the post contains a factual error
        - "manipulated_media":  # the post contains manipulated/fake/out-of-context media
        - "outdated_information":  # the post contains outdated information
        - "missing_important_context":  # the post is missing important context
        - "disputed_claim_as_fact":  # including unverified claims
        - "misinterpreted_satire":  # the post is satire that may likely be misinterpreted as fact
        - "other":  # the post contains other misleading reasons

        Example valid JSON response:
        {{
            "misleading_tags": ["factual_error", "outdated_information", "missing_important_context"]
        }}

        OTHER = 0
        FACTUAL_ERROR = 1
        MANIPULATED_MEDIA = 2
        OUTDATED_INFORMATION = 3
        MISSING_IMPORTANT_CONTEXT = 4
        DISPUTED_CLAIM_AS_FACT = 5
        MISINTERPRETED_SATIRE = 6

        The post and note are as follows:

        {post_with_context_description}

        Proposed community note:
        ```
        {note_text}
        ```
        """

    input = [{"role": "user","content": [{"type": "input_text", "text": misleading_why_tags_prompt}]}]
    while retries > 0:
        try:
            response = openai_client.responses.create(
                model="gpt-5-mini-2025-08-07",
                instructions="You are a professional fact checker.",
                input=input,
            )
            misleading_why_tags_str = response.output_text
            misleading_why_tags = json.loads(misleading_why_tags_str)["misleading_tags"]
            break
        except Exception as e:
            retries -= 1
            if retries == 0:
                raise e

    return [MisleadingTag(tag) for tag in misleading_why_tags]


def get_claimOpinion_score(
    account_name: str, note_or_refusal_str: str, post_id: str
) -> float:
    oauth_config = X_API_ACCOUNTS[account_name]

    oauth = OAuth1Session(
        client_key=oauth_config["X_API_KEY"],
        client_secret=oauth_config["X_API_KEY_SECRET"],
        resource_owner_key=oauth_config["X_ACCESS_TOKEN"],
        resource_owner_secret=oauth_config["X_ACCESS_TOKEN_SECRET"],
    )

    headers = {
        "Authorization": f"Bearer {oauth_config['X_BEARER_TOKEN']}",
        "Content-Type": "application/json",
    }
    url = "https://api.x.com/2/evaluate_note"
    payload = {
        "note_text": note_or_refusal_str,
        "post_id": post_id
    }
    
    response = oauth.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        json_response = response.json()
        claim_opinion_score = json_response['data']['claim_opinion_score']
        return claim_opinion_score
    else:
        return -100   # error