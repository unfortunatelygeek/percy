import re
from typing import List

def extract_between_tags(tag: str, string: str, strip: bool = False) -> List[str]:
    """
    Extracts content enclosed within specific XML/HTML-like tags.

    Args:
        tag (str): The tag to search for.
        string (str): The input string containing tags.
        strip (bool): Whether to strip leading/trailing whitespace from extracted content.

    Returns:
        List[str]: A list of extracted strings.
    """
    extracted = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    return [e.strip() for e in extracted] if strip else extracted
