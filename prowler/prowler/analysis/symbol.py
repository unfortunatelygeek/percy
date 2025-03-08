import jedi
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Generator, Set
from datetime import datetime
from tree_sitter import Language, Parser

# Load Tree-sitter for JS/TS
JS_PARSER = None
try:
    JS_PARSER = Parser()
    language_dir = Path('../../tree-sitter-javascript')
    if language_dir.exists():
        Language.build_library(
            'build/my-languages.so',
            [str(language_dir), str(Path('tree-sitter-python'))]
        )
        JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
        JS_PARSER.set_language(JS_LANGUAGE)
    print(f"JS was loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Tree-sitter JS/TS language: {e}")
    JS_PARSER = None

class SymbolExtractor:
    def __init__(self, repo_path: Union[str, Path]) -> None:
        self.repo_path = Path(repo_path)
        self.project = jedi.Project(self.repo_path)
        self.ignore_paths = {'/test', '_test/', '/docs', '/example', '/node_modules', '/__pycache__', '/dist'}
    
    def get_filtered_files(self, extensions: List[str]) -> Generator[Path, None, None]:
        for ext in extensions:
            for file in self.repo_path.rglob(f"*{ext}"):
                file_str = str(file).replace('\\', '/')
                if not any(ignore in file_str for ignore in self.ignore_paths):
                    yield file
    
    def extract(self, symbol_name: str, filtered_files: List[Path]) -> Optional[Dict[str, Any]]:
        if filtered_files[0].suffix == ".py":
            return self._extract_python_symbol(symbol_name, filtered_files)
        else:
            return self._extract_js_ts_symbol(symbol_name, filtered_files)
    
    def _extract_python_symbol(self, symbol_name: str, matching_files: List[Path]) -> Optional[Dict[str, Any]]:
        scripts = [jedi.Script(path=str(file), project=self.project) for file in matching_files]
        return self._search_for_symbol(symbol_name, scripts)
    
    def _search_for_symbol(self, symbol_name: str, scripts) -> Optional[Dict[str, Any]]:
        for script in scripts:
            completions = script.complete(line=1, column=0)
            for completion in completions:
                if completion.name == symbol_name:
                    return {
                        "name": symbol_name,
                        "file_path": completion.module_path,
                        "source": completion.get_line_code(),
                        "type": "python symbol"
                    }
        return None
    
    def _extract_js_ts_symbol(self, symbol_name: str, matching_files: List[Path]) -> Optional[Dict[str, Any]]:
        if JS_PARSER is None:
            print("Error: Tree-sitter JS/TS parser is not initialized.")
            return None

        for file in matching_files:
            try:
                with file.open(encoding='utf-8') as f:
                    content = f.read()
                tree = JS_PARSER.parse(content.encode('utf-8'))
                match = self._search_tree_for_symbol(symbol_name, tree, content, str(file))
                if match:
                    return match
            except Exception as e:
                print(f"Error processing {file}: {e}")
        return None
    
    def _search_tree_for_symbol(self, symbol_name: str, tree, content: str, file_path: str) -> Optional[Dict[str, Any]]:
        def traverse(node):
            if node.type in {"function_declaration", "class_declaration", "variable_declarator", "arrow_function", "method_definition"}:
                for child in node.children:
                    if child.type == "identifier" and content[child.start_byte:child.end_byte] == symbol_name:
                        return {
                            "name": symbol_name,
                            "file_path": file_path,
                            "source": content[node.start_byte:node.end_byte].strip(),
                            "type": node.type.replace("_", " ")
                        }
            for child in node.children:
                match = traverse(child)
                if match:
                    return match
            return None
        
        return traverse(tree.root_node)

class VulnerabilityExtractor:
    def __init__(self, repo_path: Union[str, Path]) -> None:
        self.repo_path = Path(repo_path)
        self.ignore_paths = {'/test', '/docs', '/node_modules', '/dist', '/__pycache__'}
    
    def get_filtered_files(self, extensions: List[str]) -> Generator[Path, None, None]:
        for ext in extensions:
            for file in self.repo_path.rglob(f"*{ext}"):
                file_str = str(file).replace('\\', '/')
                if not any(ignore in file_str for ignore in self.ignore_paths):
                    yield file
    
    def scan_vulnerabilities(self, file: Path) -> List[Dict[str, Any]]:
        vulnerabilities = []
        try:
            with file.open(encoding='utf-8') as f:
                content = f.read()
            lang = "python" if file.suffix == ".py" else "javascript"
            for vuln_type, patterns in VulnerabilityExtractor.VULNERABILITY_PATTERNS.items():
                for pattern in patterns.get(lang, []):
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        vulnerabilities.append({
                            "file": str(file),
                            "vulnerability": vuln_type,
                            "code": content[match.start():match.end()]
                        })
        except Exception as e:
            print(f"Error processing {file}: {e}")
        return vulnerabilities
    
VulnerabilityExtractor.VULNERABILITY_PATTERNS = {
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

if __name__ == "__main__":
    repo_path = "path/to/repo"
    symbol_extractor = SymbolExtractor(repo_path)
    vuln_extractor = VulnerabilityExtractor(repo_path)
    
    python_files = list(symbol_extractor.get_filtered_files([".py"]))
    js_files = list(symbol_extractor.get_filtered_files([".js", ".ts"]))
    
    for file in python_files + js_files:
        vulnerabilities = vuln_extractor.scan_vulnerabilities(file)
        if vulnerabilities:
            print(json.dumps(vulnerabilities, indent=4))