"""
Centralized configuration for the writer pipeline.
Fill in the values below before running the pipeline.
"""

# ============================================================
# API Keys
# ============================================================

OPENAI_API_KEY = ""

XAI_API_KEY = ""

# ============================================================
# X API Credentials (one entry per account)
# ============================================================
# The --account-name CLI argument selects which entry to use.
# Add as many accounts as needed; the key is the account name.

X_API_ACCOUNTS = {
    "account_1": {
        "X_API_KEY": "",
        "X_API_KEY_SECRET": "",
        "X_ACCESS_TOKEN": "",
        "X_ACCESS_TOKEN_SECRET": "",
        "X_BEARER_TOKEN": "",
    },
}

# ============================================================
# Output file paths
# ============================================================
# Directories are created automatically if they do not exist.

LOG_FILE = "output/run_log.txt"
PIPELINE_LOG_FILE = "output/pipeline_log.jsonl"
POST_IDS_WRITTEN_FILE = "output/post_ids_written.txt"
GROK_RESEARCH_OUTPUT_FILE = "output/grok_research.jsonl"

# ============================================================
# User-Agent strings for URL validation requests
# ============================================================

USER_AGENTS = []
