import argparse
import dotenv
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
import io

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

# Initialize as a list to store dictionaries with file and vulnerability info
vulnerabilities_catalog = []

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
        
    # Generate the report after completing the analysis
    generate_report()

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

def markdown_to_pdf(markdown_content, output_path):
    """Convert markdown content to PDF using reportlab."""
    try:
        # Create a PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        )
        
        file_heading_style = ParagraphStyle(
            'FileHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=8
        )
        
        normal_style = styles['Normal']
        code_style = ParagraphStyle(
            'Code',
            parent=normal_style,
            fontName='Courier',
            fontSize=9,
            leftIndent=20,
            rightIndent=20
        )
        
        # Parse the markdown content
        # For a proper markdown to PDF conversion, we need to process the markdown
        # content and convert it to ReportLab elements
        
        # This is a simplified approach - for a more comprehensive solution,
        # consider using a library specifically designed for markdown to PDF conversion
        
        story = []
        
        # Add title
        story.append(Paragraph("Security Vulnerability Report", title_style))
        story.append(Spacer(1, 12))
        
        # Split the markdown content into sections by file
        sections = markdown_content.split("---")
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split("\n")
            
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    # Main heading
                    story.append(Paragraph(line[2:], heading_style))
                elif line.startswith("## "):
                    # Subheading
                    story.append(Paragraph(line[3:], file_heading_style))
                elif line.startswith("- "):
                    # List item
                    story.append(Paragraph("• " + line[2:], normal_style))
                elif line.startswith("```"):
                    # Code block
                    code_start = i
                    code_end = None
                    
                    # Find the end of the code block
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("```"):
                            code_end = j
                            break
                    
                    if code_end:
                        code_content = "\n".join(lines[code_start + 1:code_end])
                        story.append(Paragraph(code_content, code_style))
                        i = code_end  # Skip processed lines
                else:
                    # Regular paragraph
                    if line.strip():
                        story.append(Paragraph(line, normal_style))
                
                story.append(Spacer(1, 6))
            
            # Add a spacer between sections
            story.append(Spacer(1, 12))
        
        # Build the PDF
        doc.build(story)
        log.info(f"PDF report generated: {output_path}")
        return True
        
    except Exception as e:
        log.error(f"Error converting markdown to PDF: {str(e)}")
        return False
        
def generate_report():
    """Generates markdown and PDF reports of all catalogued vulnerabilities."""
    # Generate markdown report
    markdown_path = "vulnerability_report.md"
    pdf_path = "vulnerability_report.pdf"
    
    markdown_content = "# Security Vulnerability Report\n\n"
    
    if not vulnerabilities_catalog:
        markdown_content += "No vulnerabilities were found.\n"
        log.info("No vulnerabilities found to report.")
    else:
        for entry in vulnerabilities_catalog:
            file = entry["file"]
            vulnerabilities = entry["vulnerabilities"]
            llm_response = entry["llm_response"]
            
            markdown_content += f"## File: {file}\n\n"
            markdown_content += "### Vulnerabilities:\n\n"
            
            for vuln in vulnerabilities:
                markdown_content += f"- **{vuln['type']}**: {vuln['description']}\n"
                markdown_content += f"  - Code: `{vuln['code']}`\n\n"
            
            markdown_content += "### LLM Analysis:\n\n"
            markdown_content += f"{llm_response}\n\n"
            markdown_content += "---\n\n"
    
    # Write markdown file
    with open(markdown_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)
    
    log.info(f"Markdown report saved to {markdown_path}")
    
    # Convert markdown to PDF
    if markdown_to_pdf(markdown_content, pdf_path):
        log.info(f"PDF report saved to {pdf_path}")
    else:
        log.error("Failed to generate PDF report")

def run():
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