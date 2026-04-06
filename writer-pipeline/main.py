import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List
import subprocess

from config import X_API_ACCOUNTS, LOG_FILE, POST_IDS_WRITTEN_FILE
from data_models import PostWithContext
from cnapi.get_api_eligible_posts import get_posts_eligible_for_notes
from cnapi.submit_note import submit_note
from note_writer.decide_and_write import note_writing_pipeline
from openai import BadRequestError


def _worker(
    account_name: str,
    post_with_context: PostWithContext,
    dry_run: bool = False,
    test_mode: bool = True,
) -> bool:
    """
    Fetch and try to write and submit a note for one post.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    Returns True if a note was successfully submitted, False otherwise.
    """

    print("Start research and write note...")
    note_result = note_writing_pipeline(post_with_context, account_name)

    log_strings: List[str] = [
        "-" * 20,
        f"Post: {post_with_context.post.post_id}",
        "-" * 20,
    ]
    log_strings.append(f"\nAssigned account: {account_name}")
    if note_result.context_description is not None:
        log_strings.append(
            f"\n*POST TEXT, IMAGE SUMMARIES, AND ANY QUOTED OR REPLIED-TO POST:*\n{note_result.context_description}\n"
        )
    if note_result.error is not None:
        log_strings.append(f"\n*ERROR:* {note_result.error}")
    if note_result.refusal:
        log_strings.append(f"\n*REFUSAL:* {note_result.refusal}")
    if note_result.note:
        log_strings.append(f"\n*NOTE:*\n  {note_result.note.note_text}\n")
        log_strings.append(
            f"\n*MISLEADING TAGS:*\n  {[tag.value for tag in note_result.note.misleading_tags]}\n"
        )
    if note_result.claim_opinion_score is not None:
        log_strings.append(
            f"\n*CLAIM OPINION SCORE:*  {note_result.claim_opinion_score}\n"
        )

    success = False
    if note_result.note is not None and not dry_run:
        try:
            submit_note(
                note=note_result.note,
                test_mode=test_mode,
                verbose_if_failed=False,
            )
            if test_mode:
                log_strings.append("\n*TEST MODE: SUCCESSFULLY SUBMITTED NOTE*\n")
            else:
                log_strings.append("\n*SUCCESSFULLY SUBMITTED NOTE*\n")
            success = True
        except Exception:
            log_strings.append(
                "\n*ERROR SUBMITTING NOTE*: likely we already wrote a note on this post; moving on.\n"
            )
    print("".join(log_strings) + "\n")

    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write("".join(log_strings) + "\n")

    return success


def main(
    account_name: str,
    num_posts: int = 5,
    dry_run: bool = False,
    concurrency: int = 1,
    test_mode: bool = True,
) -> int:
    """
    Get up to `num_posts` recent posts eligible for notes and write notes for them.
    If `dry_run` is True, do not submit notes to the API, just print them to the console.
    """

    post_id_written = []
    if os.path.exists(POST_IDS_WRITTEN_FILE):
        with open(POST_IDS_WRITTEN_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    post_id_written.append(line)

    print(f"Getting up to {num_posts} recent posts eligible for notes")
    fetched_posts: List[PostWithContext] = get_posts_eligible_for_notes(
        max_results=num_posts
    )
    eligible_posts = []
    for post_with_context in fetched_posts:
        if post_with_context.post.post_id not in post_id_written:
            eligible_posts.append(post_with_context)
            post_id_written.append(post_with_context.post.post_id)

    os.makedirs(os.path.dirname(POST_IDS_WRITTEN_FILE), exist_ok=True)
    with open(POST_IDS_WRITTEN_FILE, "w") as f:
        for post_id in post_id_written:
            f.write(post_id + "\n")

    print(f"Found {len(eligible_posts)} recent posts eligible for notes")
    print(
        f"  Eligible Post IDs: {', '.join([str(post_with_context.post.post_id) for post_with_context in eligible_posts])}\n"
    )
    if len(eligible_posts) == 0:
        print("No posts to process.")
        return 0

    successful_count = 0
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _worker,
                    account_name,
                    post_with_context,
                    dry_run,
                    test_mode,
                )
                for post_with_context in eligible_posts
            ]
            for future in futures:
                try:
                    if future.result():
                        successful_count += 1
                except BadRequestError as e:
                    print(f"\n*ERROR: BadRequestError occurred - {str(e)}*")
                    print("Skipping to next post...\n")
    else:
        for post_with_context in eligible_posts:
            try:
                if _worker(
                    account_name,
                    post_with_context,
                    dry_run,
                    test_mode,
                ):
                    successful_count += 1
            except BadRequestError as e:
                print(f"\n*ERROR: BadRequestError occurred - {str(e)}*")
                print("Skipping to next post...\n")
    print(f"Done. Successfully submitted {successful_count} notes for {account_name}.")
    return successful_count


def switch_account(account_name: str):
    subprocess.run(["xurl", "auth", "clear", "--all"], check=True, capture_output=True)
    config = X_API_ACCOUNTS[account_name]
    auth_cmd = [
        "xurl",
        "auth",
        "oauth1",
        "--consumer-key",
        config["X_API_KEY"],
        "--consumer-secret",
        config["X_API_KEY_SECRET"],
        "--access-token",
        config["X_ACCESS_TOKEN"],
        "--token-secret",
        config["X_ACCESS_TOKEN_SECRET"],
    ]
    subprocess.run(auth_cmd, check=True, capture_output=True)
    print(f"Switched to account: {account_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run note-writing bot once.")
    parser.add_argument(
        "--num-posts", type=int, default=10, help="Number of posts to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit notes to the API, just print them to the console",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent tasks to run",
    )
    parser.add_argument(
        "--test-mode-off",
        help="Turn off test mode",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--account-name",
        type=str,
        required=True,
        help="Account name as defined in config.py X_API_ACCOUNTS",
    )
    args = parser.parse_args()

    switch_account(args.account_name)

    test_mode = not args.test_mode_off
    main(
        account_name=args.account_name,
        num_posts=args.num_posts,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        test_mode=test_mode,
    )
