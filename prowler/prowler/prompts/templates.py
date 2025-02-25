README_SUMMARY_PROMPT = """\
Provide a very concise summary of the README.md content. Focus only on the key aspects of the project, \
such as its purpose, core functionality, and main technologies used.
"""

SYS_PROMPT = """\
You are an advanced cybersecurity assistant named Prowler, specializing in security analysis for Python applications. \
Your job is to analyze code for security vulnerabilities, identify potential exploits, and provide insights \
to developers. Maintain a structured and concise response format.
"""

INITIAL_ANALYSIS_PROMPT = """\
Analyze the provided Python file for security vulnerabilities. Identify potential security risks, \
such as SQL Injection, XSS, SSRF, RCE, AFO, LFI, IDOR. If vulnerabilities exist, \
explain why they are dangerous and how an attacker might exploit them.
"""

ANALYSIS_APPROACH = """\
Your approach should include:
1. Identifying direct vulnerabilities.
2. Checking for improper input validation.
3. Understanding the data flow to determine potential injection points.
4. Examining authentication and authorization flaws.
5. Providing a confidence score on your findings.
"""

GUIDELINES = """\
- Be specific about identified vulnerabilities.
- Avoid speculative vulnerabilities without strong reasoning.
- If a vulnerability requires additional context, request the necessary information.
- Maintain a structured format in your response.
"""

SPECIFIC_BYPASSES_AND_PROMPTS = {
    "LFI": {
        "bypasses": [
            "Using `../` sequences to escape intended directories.",
            "URL encoding techniques to evade filtering.",
            "Using null byte (%00) injection to bypass extensions."
        ],
        "prompt": """\
Investigate whether Local File Inclusion (LFI) vulnerabilities exist in the provided Python file. \
Check for unvalidated file path inputs, improper sanitization, and possible exploit paths.
"""
    },
    "RCE": {
        "bypasses": [
            "Command injection via shell execution functions.",
            "Using environment variables to bypass filters.",
            "Encoding payloads to evade detection."
        ],
        "prompt": """\
Analyze the provided Python file for potential Remote Code Execution (RCE) vulnerabilities. \
Identify areas where unsanitized user input is executed within system calls or eval functions.
"""
    },
    "XSS": {
        "bypasses": [
            "Encoding payloads to bypass filters.",
            "Using JavaScript event handlers for execution.",
            "Injecting script tags in unescaped HTML fields."
        ],
        "prompt": """\
Determine if the provided Python file is vulnerable to Cross-Site Scripting (XSS). \
Check for improper handling of user-generated content in web responses.
"""
    },
}
