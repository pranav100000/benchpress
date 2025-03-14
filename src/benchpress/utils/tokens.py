"""Utility functions for working with API tokens."""

import os
from typing import Optional


def get_hf_token() -> Optional[str]:
    """Get the Hugging Face API token from environment variables.
    
    Checks for HUGGINGFACE_API_TOKEN or HF_TOKEN environment variables.
    
    Returns:
        The token if found, None otherwise
    """
    return os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")