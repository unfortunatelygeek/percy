import re
from typing import List


def extract_between_tags(tag: str, string: str, strip: bool = False) -> List[str]:
    """
    Extracts content enclosed within specific XML/HTML-like tags.

    Supports Python, JavaScript, TypeScript (including JSX/TSX).

    Args:
        tag (str): The tag to search for.
        string (str): The input string containing tags.
        strip (bool): Whether to strip leading/trailing whitespace from extracted content.

    Returns:
        List[str]: A list of extracted strings.
    """
    extracted = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    return [e.strip() for e in extracted] if strip else extracted


def extract_js_template_strings(string: str) -> List[str]:
    """
    Extracts content from JavaScript template literals (backticks).

    Args:
        string (str): The input string containing JS template literals.

    Returns:
        List[str]: A list of extracted template literal contents.
    """
    return re.findall(r"`(.*?)`", string, re.DOTALL)


def extract_js_multiline_comments(string: str) -> List[str]:
    """
    Extracts multi-line comments from JavaScript and TypeScript.

    Args:
        string (str): The input string containing JS/TS comments.

    Returns:
        List[str]: A list of extracted multi-line comments.
    """
    return re.findall(r"/\*(.*?)\*/", string, re.DOTALL)


def extract_js_single_line_comments(string: str) -> List[str]:
    """
    Extracts single-line comments from JavaScript and TypeScript.

    Args:
        string (str): The input string containing JS/TS comments.

    Returns:
        List[str]: A list of extracted single-line comments.
    """
    return re.findall(r"//(.*)", string)
