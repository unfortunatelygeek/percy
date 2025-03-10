# import argparse
# import dotenv
# from pathlib import Path
# from prowler.analysis.repo import RepoAnalyzer, JSRepoAnalyzer
# from prowler.analysis.diff import GitDiffAnalyzer
# from prowler.analysis.js_ast.parser import analyze_code
# from prowler.analysis.symbol import SymbolExtractor
# # from prowler.analysis.js_ast.vuln_extract import VulnerabilityExtractor
# from prowler.analysis.js_ast.parser import analyze_code
# from prowler.llms.claude import Claude
# from prowler.llms.gemini import Gemini
# from prowler.llms.ollama import Ollama
# from prowler.prompts.templates import SYS_PROMPT
# from prowler.core.logger import log
# import os

# dotenv.load_dotenv()

# LLM_MODELS = {
#     "claude": lambda: Claude(model="claude-3-sonnet", base_url="https://api.anthropic.com", system_prompt=SYS_PROMPT),
#     "gemini": lambda: Gemini(model="gemini-1.5-pro", base_url="https://generativelanguage.googleapis.com", system_prompt=SYS_PROMPT),
#     "ollama": lambda: Ollama(model="codellama", base_url="http://127.0.0.1:11434/api/generate", system_prompt=SYS_PROMPT),
# }

# def initialize_llm(llm_arg: str):
#     """Initializes the LLM model based on user input."""
#     log.info(f"Initializing LLM: {llm_arg}")
#     llm_arg = llm_arg.lower()
#     if llm_arg in LLM_MODELS:
#         return LLM_MODELS[llm_arg]()
#     log.error(f"Invalid LLM argument: {llm_arg}")
#     raise ValueError(f"Invalid LLM argument: {llm_arg}")

# def generate_prompt(symbols: dict, vulnerabilities: dict, code_snippet: str) -> str:
#     log.debug("Generating analysis prompt")
#     prompt = f"{SYS_PROMPT}\n"
#     prompt += "### Extracted Symbols ###\n"
#     prompt += f"{symbols}\n\n"

#     if vulnerabilities:
#         prompt += "### Potential Vulnerabilities ###\n"
#         for vuln in vulnerabilities:
#             prompt += f"- {vuln['type']}: {vuln['description']} (Code: {vuln['code']})\n"

#     prompt += "\n### Code Snippet ###\n"
#     prompt += f"```\n{code_snippet[:5000]}\n```"  # Limiting snippet size

#     return prompt

# def analyze_repo(args):
#     """Runs full repository analysis or Git diff-based analysis."""
#     log.info("Starting repository analysis")

#     # Select RepoAnalyzer or GitDiffAnalyzer
#     if args.diff:
#         log.info("Using GitDiffAnalyzer for changed files only")
#         analyzer = GitDiffAnalyzer(args.root)
#         files_to_analyze = analyzer.analyze_git_diff()["changed_files_list"]
#     else:
#         log.info("Using full RepoAnalyzer")
#         if args.type == "python":
#             repo = RepoAnalyzer(args.root)
#             files_to_analyze = repo.get_relevant_py_files()
#         else:  # JavaScript/TypeScript
#             repo = JSRepoAnalyzer(args.root)
#             files_to_analyze = repo.get_relevant_files()

#     files_to_analyze = list(files_to_analyze)  # Convert generator to list
#     log.info(f"Total files to analyze: {len(files_to_analyze)}")    


#     # Symbol and vulnerability extraction
#     symbol_extractor = SymbolExtractor(args.root, files_to_analyze=files_to_analyze)
#     js_symbols = [
#         "eval",
#         "setTimeout",
#         "setInterval",
#         "document.write",
#         "innerHTML",
#         "localStorage.getItem",
#         "fetch",
#         "Function",
#         "child_process.exec",
#         "fs.writeFileSync",
#         "require",
#         "JSON.parse",
#         "WebSocket"
#     ]
#     py_symbols = [
#         "eval",
#         "exec",
#         "pickle.load",
#         "subprocess.Popen",
#         "os.system",
#         "shutil.rmtree",
#         "input",  
#         "open", 
#         "sqlite3.connect",
#         "yaml.load",
#         "requests.get",
#         "flask.request.args.get",
#         "django.db.connection.cursor().execute"
#     ]
#     if args.type == "python":
#         for symbol in py_symbols:
#             symbols += symbol_extractor.extract(symbol_name=symbol, files_to_analyze=files_to_analyze)
#     else:
#         for symbol in js_symbols:
#             symbols += symbol_extractor.extract(symbol_name=symbol, files_to_analyze=files_to_analyze)
#     # vuln_extractor = VulnerabilityExtractor()
    
#     for file in files_to_analyze:
#         if not os.path.exists(file):
#             print(f"⚠️ Skipping {file}: File not found.")
#             continue

#         with open(file, "r", encoding="utf-8") as f:
#             code = f.read()

#         print(f"\n=== Analyzing {file} ===")
#         graph, vulnerabilities = analyze_code(code, language="javascript")

#         if not vulnerabilities:
#             print("✅ No vulnerabilities found.")

#     # # AST Parser (if React or JS is selected)
#     # if args.type == "js":
#     #     for file in files_to_analyze:
#     #         file_path = Path(file)
#     #         try:
#     #             code = file_path.read_text(encoding="utf-8")
#     #             ast_graph, detected_vulns = analyze_code(code, language="javascript")

#     #             if detected_vulns:
#     #                 log.warning(f"Vulnerabilities detected in {file}")
#     #                 for vuln in detected_vulns:
#     #                     log.warning(f" - {vuln['type']}: {vuln['description']} (Code: {vuln['code']})")

#     #             # Extract symbols (e.g., API routes, DB queries)
#     #             symbols = symbol_extractor.extract(symbol_name="your_symbol", files_to_analyze=[file])

#             # Pass vulnerabilities to LLM for fixes
#             llm = initialize_llm(args.llm)
#             prompt = generate_prompt(symbols, vulnerabilities, code)
#             # response = llm.chat(prompt)
#             response = llm.chat(prompt) if prompt else None
#             if response:
#                 log.info(f"LLM Response for {file}", response=response)
#             else:
#                 log.warning(f"LLM returned an empty response for {file}")
#             log.info(f"LLM Response for {file}", response=response)
#         except Exception as e:
#             log.error(f"Error analyzing {file}: {str(e)}")

# def run():
#     """Main function to parse CLI args and start analysis."""
#     parser = argparse.ArgumentParser(description="Analyze a repository for vulnerabilities and structure changes.")
#     parser.add_argument("-r", "--root", type=str, required=True, help="Path to project root")
#     parser.add_argument("-a", "--analyze", type=str, help="Specific file or directory to analyze")
#     parser.add_argument("-l", "--llm", type=str, choices=LLM_MODELS.keys(), default="claude", help="LLM model")
#     parser.add_argument("-t", "--type", type=str, choices=["python", "js"], required=True, help="Project type (python or js)")
#     parser.add_argument("-d", "--diff", action="store_true", help="Analyze only git diff changes")
    
#     args = parser.parse_args()
#     log.info(f"Arguments received: {args}")
    
#     analyze_repo(args)

# if __name__ == "__main__":
#     run()

import argparse
import dotenv
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

from prowler.analysis.repo import RepoAnalyzer, JSRepoAnalyzer
from prowler.analysis.diff import GitDiffAnalyzer
from prowler.analysis.js_ast.parser import analyze_code
from prowler.analysis.symbol import SymbolExtractor
from prowler.llms.claude import Claude
from prowler.llms.gemini import Gemini
from prowler.llms.ollama import Ollama
from prowler.prompts.templates import SYS_PROMPT
from prowler.core.logger import log

# Load environment variables
dotenv.load_dotenv()

# Define LLM model configurations
LLM_MODELS = {
    "claude": lambda: Claude(
        model="claude-3-sonnet", 
        base_url="https://api.anthropic.com", 
        system_prompt=SYS_PROMPT
    ),
    "gemini": lambda: Gemini(
        model="gemini-1.5-pro", 
        base_url="https://generativelanguage.googleapis.com", 
        system_prompt=SYS_PROMPT
    ),
    "ollama": lambda: Ollama(
        model="codellama", 
        base_url="http://127.0.0.1:11434/api/generate", 
        system_prompt=SYS_PROMPT
    ),
}

# Define common risky symbols to search for
JS_SYMBOLS = [
    "eval", "setTimeout", "setInterval", "document.write", "innerHTML",
    "localStorage.getItem", "fetch", "Function", "child_process.exec",
    "fs.writeFileSync", "require", "JSON.parse", "WebSocket"
]

PY_SYMBOLS = [
    "eval", "exec", "pickle.load", "subprocess.Popen", "os.system",
    "shutil.rmtree", "input", "open", "sqlite3.connect", "yaml.load",
    "requests.get", "flask.request.args.get", "django.db.connection.cursor().execute"
]

vulnerabilities_catalog = {}

def initialize_llm(llm_arg: str):
    """Initializes the LLM model based on user input."""
    log.info(f"Initializing LLM: {llm_arg}")
    llm_arg = llm_arg.lower()
    
    if llm_arg in LLM_MODELS:
        return LLM_MODELS[llm_arg]()
    
    log.error(f"Invalid LLM argument: {llm_arg}")
    raise ValueError(f"Invalid LLM argument: {llm_arg}")

def generate_prompt(symbols: dict, vulnerabilities: dict, code_snippet: str) -> str:
    """Generate analysis prompt for LLM with extracted symbols and vulnerabilities."""
    log.debug("Generating analysis prompt")
    prompt = f"{SYS_PROMPT}\n"
    prompt += "### Extracted Symbols ###\n"
    prompt += f"{symbols}\n\n"

    if vulnerabilities:
        prompt += "### Potential Vulnerabilities ###\n"
        for vuln in vulnerabilities:
            prompt += f"- {vuln['type']}: {vuln['description']} (Code: {vuln['code']})\n"

    prompt += "\n### Code Snippet ###\n"
    # Limiting snippet size to prevent token overflow
    prompt += f"```\n{code_snippet[:5000]}\n```"

    return prompt

def analyze_repo(args):
    """Runs full repository analysis or Git diff-based analysis."""
    log.info("Starting repository analysis")

    # Get files to analyze based on analysis type
    files_to_analyze = get_files_to_analyze(args)
    files_to_analyze = list(files_to_analyze)  # Convert generator to list
    log.info(f"Total files to analyze: {len(files_to_analyze)}")

    # Initialize symbol extractor
    symbol_extractor = SymbolExtractor(args.root, files_to_analyze=files_to_analyze)
    symbols = []

    # Extract symbols based on project type
    if args.type == "python":
        for symbol in PY_SYMBOLS:
            symbols += symbol_extractor.extract(symbol_name=symbol, files_to_analyze=files_to_analyze)
    else:
        for symbol in JS_SYMBOLS:
            symbols += symbol_extractor.extract(symbol_name=symbol, files_to_analyze=files_to_analyze)

    # Process each file
    for file in files_to_analyze:
        analyze_file(file, symbol_extractor, symbols, args)

def get_files_to_analyze(args):
    """Get the list of files to analyze based on provided arguments."""
    if args.diff:
        log.info("Using GitDiffAnalyzer for changed files only")
        analyzer = GitDiffAnalyzer(args.root)
        return analyzer.analyze_git_diff()["changed_files_list"]
    else:
        log.info("Using full RepoAnalyzer")
        if args.type == "python":
            repo = RepoAnalyzer(args.root)
            return repo.get_relevant_py_files()
        else:  # JavaScript/TypeScript
            repo = JSRepoAnalyzer(args.root)
            return repo.get_relevant_files()

def analyze_file(file, symbol_extractor, symbols, args):
    """Analyze a single file for vulnerabilities."""
    if not os.path.exists(file):
        print(f"⚠️ Skipping {file}: File not found.")
        return

    try:
        with open(file, "r", encoding="utf-8") as f:
            code = f.read()

        print(f"\n=== Analyzing {file} ===")
        
        # Convert Path object to string before using endswith
        file_str = str(file)
        
        # Analyze code using appropriate parser
        if file_str.endswith(('.js', '.jsx', '.ts', '.tsx')):
            graph, vulnerabilities = analyze_code(code, language="javascript")
        else:
            # Placeholder for Python analysis (not implemented in the original code)
            graph, vulnerabilities = None, []

        if not vulnerabilities:
            print("✅ No vulnerabilities found.")
            return

        # Initialize LLM and generate analysis
        llm = initialize_llm(args.llm)
        prompt = generate_prompt(symbols, vulnerabilities, code)
        
        if not prompt:
            log.warning(f"Empty prompt generated for {file}")
            return
            
        response = llm.chat(prompt)
        
        if response:
            log.info(f"LLM Response for {file}", response=response)
            vulnerabilities_catalog.append({
                "file": file,
                "vulnerabilities": vulnerabilities,
                "llm_response": response
            })
        else:
            log.warning(f"LLM returned an empty response for {file}")
    
    except Exception as e:
        log.error(f"Error analyzing {file}: {str(e)}")
        
def generate_report():
    """Generates a report of all catalogued vulnerabilities."""
    report_path = "vulnerability_report.txt"
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("=== Vulnerability Report ===\n\n")
        for file, vulnerabilities in vulnerabilities_catalog.items():
            report_file.write(f"File: {file}\n")
            for vuln in vulnerabilities:
                report_file.write(f"- {vuln['type']}: {vuln['description']} (Code: {vuln['code']})\n")
            report_file.write("\n")
    log.info(f"Vulnerability report saved to {report_path}")

def run():
    """Main function to parse CLI args and start analysis."""
    parser = argparse.ArgumentParser(description="Analyze a repository for vulnerabilities and structure changes.")
    parser.add_argument("-r", "--root", type=str, required=True, help="Path to project root")
    parser.add_argument("-a", "--analyze", type=str, help="Specific file or directory to analyze")
    parser.add_argument("-l", "--llm", type=str, choices=LLM_MODELS.keys(), default="claude", help="LLM model")
    parser.add_argument("-t", "--type", type=str, choices=["python", "js"], required=True, help="Project type (python or js)")
    parser.add_argument("-d", "--diff", action="store_true", help="Analyze only git diff changes")
    
    args = parser.parse_args()
    log.info(f"Arguments received: {args}")
    
    try:
        analyze_repo(args)
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt detected. Generating report before exit...")
        generate_report()
        sys.exit(1)

if __name__ == "__main__":
    run()