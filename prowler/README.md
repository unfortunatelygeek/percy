# Prowler: A Comprehensive LLM-Powered Vulnerability Detection System

## Overview  
Prowler is a powerful system designed to analyze source code for vulnerabilities using Large Language Models (LLMs) and code parsing techniques. It integrates multiple LLMs (e.g., Claude, Gemini, Ollama) to understand code contextually and detect security issues that traditional static analysis tools might miss.  

## Project Status  
**In Progress**  

## Core Functionality  

### Repository Analysis  
- Prowler scans repositories to identify relevant Python files.  
- Extracts symbols (functions, classes, variables) for deeper codebase understanding.  

### LLM-Driven Vulnerability Detection  
- Sends code snippets to LLMs with structured prompts.  
- Detects vulnerabilities like injection attacks, insecure dependencies, and access control flaws.  
- Logs responses for review.  

### Error Handling & Rate Limits  
- Manages API failures gracefully (e.g., rate limits, connectivity issues).  
- Supports interchangeable LLMs to balance cost, accuracy, and availability.  

## Installation  
Prowler uses **Poetry** for dependency management. To install:  

```sh
poetry install
```
### Usage
Run Prowler using the following command:

```sh
poetry run python main.py
```
#### Available options:
```bash
  -h, --help            Show this help message and exit  
  -r ROOT, --root ROOT  Path to the root directory of the project  
  -a ANALYZE, --analyze ANALYZE  
                        Specific path or file within the project to analyze  
  -l {claude,gemini,ollama}, --llm {claude,gemini,ollama}  
                        LLM client to use 
  -v, --verbosity       Increase output verbosity (-v for INFO, -vv for DEBUG)  

```

## To-Do

Upcoming features include:

- Tree-Sitter Integration for multi-language support.
- AST-Based Analysis to improve structured vulnerability detection.
- Zero-Day Vulnerability Detection Enhancements for better anomaly detection.
- Automated Unit Test Generation to prevent security regressions.

### External Documentation & Resources
- [VulnHuntr](https://github.com/protectai/vulnhuntr)