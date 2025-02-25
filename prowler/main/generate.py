import json
import os
from pathlib import Path
from LLMs import Claude, ChatGPT, Ollama

def load_report(report_path: str) -> dict:
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Vulnerability report not found: {report_path}")
    with open(report_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def initialize_llm():
    llm_type = os.getenv("LLM_TYPE", "gpt")
    if llm_type == "claude":
        return Claude(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"), base_url=os.getenv("ANTHROPIC_BASE_URL"))
    elif llm_type == "gpt":
        return ChatGPT(model=os.getenv("OPENAI_MODEL", "gpt-4o"), base_url=os.getenv("OPENAI_BASE_URL"))
    elif llm_type == "ollama":
        return Ollama(model=os.getenv("OLLAMA_MODEL", "llama3"), base_url=os.getenv("OLLAMA_BASE_URL"))
    else:
        raise ValueError("Unsupported LLM type")

def generate_unit_test(llm, vulnerability: dict) -> str:
    prompt = f'''
A security vulnerability was in this codebase. Generate a unit test that would have caught this issue.

### Context:
- Vulnerability type: {vulnerability["vulnerability_type"]}
- Affected file: {vulnerability["file_path"]}
- Relevant function/method: {vulnerability["function"]}
- Vulnerability description: {vulnerability["description"]}
- Exploitation scenario: {vulnerability["exploit_scenario"]}

### Task:
Generate a **Python unit test** using `pytest` or `unittest` to detect this vulnerability. The test should:
1. Replicate the conditions under which the vulnerability is triggered.
2. Assert that the function behaves **securely** and does not allow exploitation.
3. Include necessary mock inputs or dependencies.

Ensure the unit test properly fails in a vulnerable state and passes when the issue is mitigated.
'''
    
    return llm.chat(prompt)

def main():
    report_path = os.getenv("REPORT", "report.json")
    vulnerability = load_report(report_path)
    llm = initialize_llm()
    unit_test = generate_unit_test(llm, vulnerability)
    
    output_path = Path("generated_test.py")
    with output_path.open("w", encoding="utf-8") as f:
        f.write(unit_test)
    
    print(f"Unit test generated and saved to {output_path}")

if __name__ == "__main__":
    main()

