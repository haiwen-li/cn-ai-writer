# Writer Pipeline

Automated Community Notes writer for X (Twitter). This pipeline fetches posts eligible for Community Notes, researches their claims using Grok, drafts and evaluates notes using GPT, and submits them via the X API.

This code accompanies the paper: *"AI Fact-Checking in the Wild: A Field Evaluation of LLM-Written Community Notes on X"*.

## Pipeline Overview

For each eligible post, the pipeline runs the following steps:

1. **Fetch** posts eligible for Community Notes from the X API.
2. **Research** the post's claims using Grok (xAI) with web and X search tools.
3. **Decide** whether a note is warranted based on the research evidence (GPT).
4. **Write** a concise (<280 char) Community Note grounded in cited evidence (GPT).
5. **Score** the draft note using X's `evaluate_note` endpoint (claim-opinion score).
6. **Validate** that all URLs in the note exist in the evidence and are reachable.
7. **Tag** the note with misleading-reason categories (GPT).
8. **Submit** the final note to the Community Notes API.

## File Descriptions

```
writer-pipeline/
├── config.py                          # All configuration: API keys, file paths, user agents
├── main.py                            # CLI entry point; fetches posts, orchestrates workers
├── data_models.py                     # Pydantic models (Post, Note, NoteResult, etc.)
├── requirements.txt                   # Python dependencies
│
├── cnapi/                             # X / Community Notes API helpers
│   ├── get_api_eligible_posts.py      # Fetches and parses posts eligible for notes
│   ├── submit_note.py                 # Submits a note via POST /2/notes
│   └── xurl_util.py                   # Subprocess wrapper for the xurl CLI tool
│
└── note_writer/                       # Note research, drafting, and evaluation
    ├── grok_research.py               # Grok-based claim research (web + X search)
    ├── decide_and_write.py            # Decision logic, note generation, full pipeline
    ├── writer_util.py                 # Post formatting, image description, misleading tags, claim-opinion scoring
    └── url_evaluator.py               # URL extraction and validation for drafted notes
```

## Prerequisites

- **Python 3.10+**
- **[xurl](https://github.com/xdevplatform/xurl)** — CLI tool for authenticated X API requests. Must be installed and available on your `PATH`.
- **API keys** for OpenAI, xAI (Grok), and X (OAuth 1.0a credentials). See [Configuration](#configuration).

## Setup

```bash
cd writer-pipeline
pip install -r requirements.txt
```

## Configuration

Open **`config.py`** and fill in every empty string:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (used for note drafting, tagging, image description) |
| `XAI_API_KEY` | xAI API key (used for Grok research) |
| `X_API_ACCOUNTS` | Dict of X OAuth credentials keyed by account name |

Output file paths (`LOG_FILE`, `PIPELINE_LOG_FILE`, etc.) have sensible defaults under `output/` and can be changed as needed.

## Usage

```bash
python main.py --account-name <ACCOUNT_NAME> [OPTIONS]
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--account-name` | *(required)* | Account name matching a key in `config.py` `X_API_ACCOUNTS` |
| `--num-posts` | `10` | Number of eligible posts to fetch and process |
| `--dry-run` | off | Run the full pipeline but skip submitting notes to the API |
| `--concurrency` | `1` | Number of posts to process in parallel |
| `--test-mode-off` | off | Submit real (non-test) notes; default is test mode |

### Examples

Dry run on 5 posts (no submission):

```bash
python main.py --account-name account_1 --num-posts 5 --dry-run
```

Submit test-mode notes for 10 posts:

```bash
python main.py --account-name account_1 --num-posts 10
```

Submit real notes with concurrency:

```bash
python main.py --account-name account_1 --num-posts 20 --concurrency 4 --test-mode-off
```
