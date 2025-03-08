import argparse
import structlog
import dotenv
from pathlib import Path

from prowler.analysis.repo import RepoAnalyzer, JSRepoAnalyzer
from prowler.analysis.symbol import SymbolExtractor
from prowler.llms.claude import Claude
from prowler.llms.gemini import Gemini
from prowler.llms.ollama import Ollama
from prowler.prompts.templates import SYS_PROMPT
from prowler.core.logger import log

dotenv.load_dotenv()

def initialize_llm(llm_arg: str):
    """Initializes the LLM model based on the argument."""
    log.info(f"Initializing LLM: {llm_arg}")
    llm_arg = llm_arg.lower()
    if llm_arg == 'claude':
        return Claude(model="claude-3-sonnet", base_url="https://api.anthropic.com", system_prompt=SYS_PROMPT)
    elif llm_arg == 'gemini':
        return Gemini(model="gemini-1.5-pro", base_url="https://generativelanguage.googleapis.com", system_prompt=SYS_PROMPT)
    elif llm_arg == 'ollama':
        return Ollama(model="llama3", base_url="http://127.0.0.1:11434/api/generate", system_prompt=SYS_PROMPT)
    else:
        log.error(f"Invalid LLM argument: {llm_arg}")
        raise ValueError(f"Invalid LLM argument: {llm_arg}")

def generate_prompt(content: str) -> str:
    """Generates an analysis prompt for the LLM."""
    log.debug("Generating analysis prompt")
    return f"{SYS_PROMPT}\nAnalyze the following code:\n{content}"

def run():
    log.info("Starting repository analysis")
    parser = argparse.ArgumentParser(description='Analyze a repository for vulnerabilities.')
    parser.add_argument('-r', '--root', type=str, required=True, help='Path to project root')
    parser.add_argument('-a', '--analyze', type=str, help='Specific file or directory to analyze')
    parser.add_argument('-l', '--llm', type=str, choices=['claude', 'gemini', 'ollama'], default='claude', help='LLM model')
    parser.add_argument('-t', '--type', type=str, choices=['python', 'js'], required=True, help='Project type (python or js)')
    args = parser.parse_args()
    
    log.info(f"Arguments received: {args}")
    
    # Choose the appropriate repo analyzer based on project type
    if args.type == 'python':
        log.info("Using Python RepoAnalyzer")
        repo = RepoAnalyzer(args.root)
        files_to_analyze = repo.get_relevant_py_files()
    else:  # JavaScript/TypeScript
        log.info("Using JavaScript RepoAnalyzer")
        repo = JSRepoAnalyzer(args.root)
        files_to_analyze = repo.get_relevant_files()
    
    symbol_extractor = SymbolExtractor(args.root)
    llm = initialize_llm(args.llm)
    
    for file in files_to_analyze:
        log.info(f"Analyzing {file}")
        try:
            content = Path(file).read_text(encoding='utf-8')
            prompt = generate_prompt(content)
            response = llm.chat(prompt)
            log.info(f"Analysis complete for {file}", response=response)
        except Exception as e:
            log.error(f"Error analyzing {file}: {str(e)}")

if __name__ == '__main__':
    run()
