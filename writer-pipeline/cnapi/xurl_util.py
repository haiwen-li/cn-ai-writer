import json
import subprocess
from typing import Any, Dict, List, Optional


class XurlError(Exception):
    """Exception raised when xurl command fails."""

    def __init__(self, message: str, error_response: Optional[Dict[str, Any]] = None,
                 returncode: int = 1, stderr: str = ""):
        super().__init__(message)
        self.error_response = error_response or {}
        self.returncode = returncode
        self.stderr = stderr

    @property
    def detail(self) -> Optional[str]:
        """Get the 'detail' field from the error response."""
        return self.error_response.get('detail')

    @property
    def status(self) -> Optional[int]:
        """Get the 'status' field from the error response."""
        return self.error_response.get('status')

    @property
    def title(self) -> Optional[str]:
        """Get the 'title' field from the error response."""
        return self.error_response.get('title')


def run_xurl(cmd: List[str], verbose_if_failed: bool = False) -> Dict[str, Any]:
    """
    Run `xurl` and return its JSON stdout as a Python dict.
    Currently extremely simple without any retry logic.

    Raises:
        XurlError: If the xurl command fails, with parsed error details if available.
    """
    try:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        error_response = None

        # Try to parse the error response from stdout
        if exc.stdout:
            try:
                error_response = json.loads(exc.stdout)
            except json.JSONDecodeError:
                # stdout might contain JSON followed by other text
                # Try to extract just the JSON part
                try:
                    # Find the first '{' and try to parse from there
                    start_idx = exc.stdout.find('{')
                    if start_idx != -1:
                        # Find the matching closing brace
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(exc.stdout)):
                            if exc.stdout[i] == '{':
                                brace_count += 1
                            elif exc.stdout[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        json_str = exc.stdout[start_idx:end_idx]
                        error_response = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    pass

        if verbose_if_failed:
            print(f"\n[ xurl failed with exit code {exc.returncode} ]", flush=True)
            if exc.stdout:
                print("── stdout ──")
                print(exc.stdout, end="", flush=True)
            if exc.stderr:
                print("── stderr ──")
                print(exc.stderr, end="", flush=True)

        # Create a meaningful error message
        if error_response and 'detail' in error_response:
            message = error_response['detail']
        else:
            message = f"xurl command failed with exit code {exc.returncode}"

        raise XurlError(
            message=message,
            error_response=error_response,
            returncode=exc.returncode,
            stderr=exc.stderr
        )

    return json.loads(completed.stdout)
