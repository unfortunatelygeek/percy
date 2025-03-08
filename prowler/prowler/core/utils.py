import jedi
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Generator, Set, Tuple
from datetime import datetime
import tree_sitter
from tree_sitter import Language, Parser

# Load Tree-sitter for JS/TS
JS_PARSER = None
try:
    # Create parser instance first
    JS_PARSER = Parser()
    
    # Build language library - corrected implementation
    language_dir = Path('tree-sitter-javascript')
    if language_dir.exists():
        Language.build_library(
            'build/my-languages.so',
            [str(language_dir), str(Path('tree-sitter-python'))]
        )
        # Load the language
        JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
        # Set the language for the parser
        JS_PARSER.set_language(JS_LANGUAGE)
except Exception as e:
    print(f"Warning: Could not load Tree-sitter JS/TS language: {e}")
    JS_PARSER = None  # Fallback in case JS/TS parsing fails


class VulnerabilityExtractor:
    """
    Scans code for potential security vulnerabilities:
    - Local File Include (LFI)
    - Arbitrary File Overwrite (AFO)
    - Remote Code Execution (RCE)
    - Cross-Site Scripting (XSS)
    - SQL Injection (SQLI)
    - Server-Side Request Forgery (SSRF)
    - Insecure Direct Object Reference (IDOR)
    """
    
    VULNERABILITY_PATTERNS = {
        "LFI": {
            "python": [
                r"open\s*\(\s*.*?['\"].*?\+.*?['\"]",
                r"open\s*\(\s*[^'\",]*?(?:user|input|request|param)",
                r"(?:read|include|require)(?:file)?\s*\(\s*.*?(?:user|input|request|param)",
                r"os\.path\.join\(.*?(?:user|input|request|param)"
            ],
            "javascript": [
                r"(?:require|fs\.read(?:File|Sync)|fs\.open).*?\(.*?(?:req\.params|req\.query|req\.body)",
                r"(?:require|fs\.read(?:File|Sync)|fs\.open).*?\(.*?\+.*?\)",
                r"path\.(?:join|resolve)\(.*?(?:req\.(?:params|query|body)|user|input)"
            ]
        },
        "AFO": {
            "python": [
                r"(?:write|create|save|store).*?(?:file|to).*?(?:user|input|request|param)",
                r"open\s*\(\s*.*?,\s*['\"]\s*[wa]\s*['\"].*?(?:user|input|request|param)",
                r"(?:shutil|os)\.(?:copy|move|rename).*?(?:user|input|request|param)"
            ],
            "javascript": [
                r"fs\.(?:write|append|create|copy|rename).*?\(.*?(?:req\.(?:params|query|body)|user|input)",
                r"(?:writeFile|writeFileSync|appendFile|appendFileSync).*?(?:req\.(?:params|query|body)|user|input)"
            ]
        },
        "RCE": {
            "python": [
                r"(?:eval|exec|subprocess\.(?:call|Popen|run)|os\.(?:system|popen|spawn|exec))\s*\(",
                r"(?:eval|exec|subprocess\.(?:call|Popen|run)|os\.(?:system|popen|spawn|exec)).*?(?:user|input|request|param)",
                r"__import__\s*\(\s*(?:user|input|request|param)"
            ],
            "javascript": [
                r"(?:eval|Function|setTimeout|setInterval|new\s+Function)\s*\(",
                r"(?:eval|Function|setTimeout|setInterval|new\s+Function).*?(?:req\.(?:params|query|body)|user|input)",
                r"(?:exec|spawn|execFile|fork)\s*\(.*?(?:req\.(?:params|query|body)|user|input)",
                r"child_process\.(?:exec|spawn|execFile|fork)"
            ]
        },
        "XSS": {
            "python": [
                r"(?:response|html|template).*?\+.*?(?:user|input|request|param)",
                r"(?:render|render_template).*?(?:user|input|request|param)",
                r"(?:send|write|output).*?(?:html|response).*?(?:user|input|request|param)"
            ],
            "javascript": [
                r"(?:innerHTML|outerHTML|document\.write|insertAdjacentHTML)\s*=",
                r"(?:innerHTML|outerHTML|document\.write|insertAdjacentHTML).*?(?:req\.(?:params|query|body)|user|input)",
                r"(?:dangerouslySetInnerHTML|render\().*?(?:req\.(?:params|query|body)|user|input)",
                r"res\.send\(.*?(?:req\.(?:params|query|body)|user|input)"
            ]
        },
        "SQLI": {
            "python": [
                r"(?:execute|cursor\.execute|query|raw|cursor|cursor\.executemany)\s*\(.*?\+",
                r"(?:execute|cursor\.execute|query|raw|cursor|cursor\.executemany).*?(?:user|input|request|param)",
                r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP).*?\+.*?(?:user|input|request|param)",
                r"(?:connection|db|cursor|conn)\.(?:execute|query)"
            ],
            "javascript": [
                r"(?:query|db\.query|connection\.query|sql\.query).*?(?:req\.(?:params|query|body)|user|input)",
                r"(?:query|db\.query|connection\.query|sql\.query).*?\+.*?(?:req\.(?:params|query|body)|user|input)",
                r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP).*?\+.*?(?:req\.(?:params|query|body)|user|input)"
            ]
        },
        "SSRF": {
            "python": [
                r"(?:requests|urllib|http|httplib|urllib2)\.(?:get|post|request|urlopen|Request)",
                r"(?:requests|urllib|http|httplib|urllib2)\.(?:get|post|request|urlopen|Request).*?(?:user|input|request|param)",
                r"(?:fetch|download|get).*?(?:url|uri|endpoint|api).*?(?:user|input|request|param)"
            ],
            "javascript": [
                r"(?:fetch|axios|http\.get|https\.get|request|superagent)",
                r"(?:fetch|axios|http\.get|https\.get|request|superagent).*?(?:req\.(?:params|query|body)|user|input)",
                r"new\s+URL\s*\(.*?(?:req\.(?:params|query|body)|user|input)"
            ]
        },
        "IDOR": {
            "python": [
                r"(?:get|find|select|query).*?(?:by|with|where).*?(?:id|uuid|record|object).*?(?:user|input|request|param)",
                r"(?:model|entity|record|object)\.(?:get|find|load)\s*\(\s*(?:user|input|request|param)",
                r"(?:user|account|profile|record).*?(?:id|uuid).*?(?:user|input|request|param)"
            ],
            "javascript": [
                r"(?:findById|getById|selectById|queryById|ObjectId).*?(?:req\.(?:params|query|body)|user|input)",
                r"(?:\.find\(|\.get\(|\.select\(|\.query\().*?(?:req\.(?:params|query|body)\.id|user|input)",
                r"(?:user|account|profile|record).*?(?:id|uuid).*?(?:req\.(?:params|query|body)|user|input)"
            ]
        }
    }
    
    def __init__(self, repo_path: Union[str, Path]) -> None:
        self.repo_path = Path(repo_path)
        self.ignore_paths = {'/test', '_test/', '/docs', '/example', '/node_modules', '/__pycache__', '/dist'}
        self.vulnerabilities: Dict[str, List[Dict[str, Any]]] = {vuln_type: [] for vuln_type in self.VULNERABILITY_PATTERNS.keys()}
    
    def get_filtered_files(self, extensions: List[str]) -> Generator[Path, None, None]:
        """Get files with specified extensions, excluding paths in ignore_paths."""
        for ext in extensions:
            for file in self.repo_path.rglob(f"*{ext}"):
                file_str = str(file).replace('\\', '/')
                if not any(ignore in file_str for ignore in self.ignore_paths):
                    yield file
    
    def scan_repo(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan the entire repository for potential vulnerabilities.
        Returns a dictionary mapping vulnerability types to lists of findings.
        """
        # Scan Python files
        py_files = list(self.get_filtered_files(['.py']))
        for file in py_files:
            self._scan_file(file, "python")
        
        # Scan JavaScript files
        js_files = list(self.get_filtered_files(['.js', '.ts', '.jsx', '.tsx']))
        for file in js_files:
            self._scan_file(file, "javascript")
        
        return self.vulnerabilities
    
    def _scan_file(self, file_path: Path, language: str) -> None:
        """
        Scan a single file for potential vulnerabilities.
        """
        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for vuln_type, patterns in self.VULNERABILITY_PATTERNS.items():
                if language in patterns:
                    for pattern in patterns[language]:
                        matches = self._find_pattern_in_content(content, pattern)
                        for match in matches:
                            match_start, match_end = match
                            
                            # Find line number and extract context (5 lines before and after)
                            line_num = content[:match_start].count('\n') + 1
                            start_line = max(0, line_num - 6)
                            end_line = min(len(lines), line_num + 5)
                            
                            # Get the code snippet with context
                            code_snippet = '\n'.join(lines[start_line:end_line])
                            
                            # Add to vulnerabilities
                            self.vulnerabilities[vuln_type].append({
                                'file_path': str(file_path.relative_to(self.repo_path)),
                                'line': line_num,
                                'match': content[match_start:match_end],
                                'snippet': code_snippet,
                                'language': language
                            })
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    def _find_pattern_in_content(self, content: str, pattern: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of a regex pattern in the content.
        Returns a list of (start, end) positions.
        """
        matches = []
        for match in re.finditer(pattern, content):
            matches.append((match.start(), match.end()))
        return matches
    
    def save_vulnerabilities_to_file(self, output_file: str = None) -> str:
        """
        Save extracted vulnerability snippets to a file.
        Each vulnerability type is clearly marked.
        Returns the path to the saved file.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"vulnerabilities_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for vuln_type, findings in self.vulnerabilities.items():
                if findings:
                    f.write(f"\n{'=' * 80}\n")
                    f.write(f"VULNERABILITY TYPE: {vuln_type}\n")
                    f.write(f"{'=' * 80}\n\n")
                    
                    for i, finding in enumerate(findings, 1):
                        f.write(f"Finding #{i}:\n")
                        f.write(f"File: {finding['file_path']}\n")
                        f.write(f"Line: {finding['line']}\n")
                        f.write(f"Language: {finding['language']}\n")
                        f.write(f"Matched pattern: {finding['match']}\n\n")
                        f.write("Code Snippet:\n")
                        f.write(f"```{finding['language']}\n{finding['snippet']}\n```\n\n")
                        f.write(f"{'-' * 40}\n\n")
        
        # Also save as JSON for programmatic access
        json_output_file = output_file.replace('.txt', '.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.vulnerabilities, f, indent=2)
        
        return output_file
    
    def extract_context(self, file_path: Path, line_num: int, context_lines: int = 5) -> str:
        """
        Extract a code snippet with context from a file.
        """
        try:
            with file_path.open('r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_line = max(0, line_num - context_lines - 1)
            end_line = min(len(lines), line_num + context_lines)
            
            return ''.join(lines[start_line:end_line])
        except Exception as e:
            print(f"Error extracting context from {file_path}: {e}")
            return ""


# Example usage:
# extractor = VulnerabilityExtractor("/path/to/repo")
# vulnerabilities = extractor.scan_repo()
# output_file = extractor.save_vulnerabilities_to_file()
# print(f"Vulnerabilities saved to {output_file}")