import re
from pathlib import Path
from typing import List, Generator
from prowler.core.logger import log

class RepoAnalyzer:
    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.to_exclude = {'/setup.py', '/test', '/example', '/docs', '/site-packages', '.venv', 'virtualenv', '/dist'}
        self.file_names_to_exclude = {'test_', 'conftest', '_test.py'}

        self.patterns = [
            r'@app\.route\(.*?\)',  # Flask routes
            r'@router\.(?:get|post|put|delete|patch|options|head|trace)\(.*?\)',  # FastAPI
            r'@blueprint\.route\(.*?\)',  # Flask blueprints
            r'@websocket\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',  # WebSockets
            r'websockets\.serve\(.*?\)',
            r'async\sdef\s\w+\(.*?request',  # Async functions
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.patterns]
        log.info("RepoAnalyzer initialized", repo_path=str(self.repo_path))

    def get_relevant_py_files(self) -> Generator[Path, None, None]:
        log.info("Scanning for relevant Python files...")
        for file in self.repo_path.rglob("*.py"):
            file_str = str(file).replace('\\', '/').lower()

            if any(exclude in file_str for exclude in self.to_exclude) or any(fn in file.name for fn in self.file_names_to_exclude):
                continue

            log.debug("Found relevant Python file", file=str(file))
            yield file

    def get_network_related_files(self, files: List[Path]) -> Generator[Path, None, None]:
        log.info("Scanning for network-related Python files...")
        for file in files:
            try:
                content = file.read_text(encoding='utf-8', errors='ignore')
                if any(pattern.search(content) for pattern in self.compiled_patterns):
                    log.debug("Network-related file detected", file=str(file))
                    yield file
            except Exception as e:
                # log.error("Error processing file", file=str(file), error=str(e))
                continue

class JSRepoAnalyzer:
    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.to_exclude = {
            "/node_modules", "/test", "/example", "/docs", "/dist", "/coverage", "/build", "/.next", "/out", "/.turbo"
        }
        self.file_names_to_exclude = {'.spec.js', '.spec.ts', '.test.js', '.test.ts'}

        self.network_patterns = [
            r'app\.use\(', r'router\.(?:get|post|put|delete|patch|options|head)\(', r'io\.on\("connection"',
            r'fetch\(', r'axios\.', r'new WebSocket\(', r'ws\.Server\(', r'http\.createServer',
            r'express\(\)', r'require\([\'\"]express[\'\"]', r'import\s+.*\s+from\s+[\'\"]express[\'\"]',
            r'import\s+.*\s+from\s+[\'\"]axios[\'\"]', r'import\s+.*\s+from\s+[\'\"]http[\'\"]',
            r'\.listen\(\d+', r'createServer\(', r'socket\.io', r'\.addEventListener\([\'\"]fetch[\'\"]'
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.network_patterns]
        log.info("JSRepoAnalyzer initialized", repo_path=str(self.repo_path))

    def get_relevant_files(self) -> Generator[Path, None, None]:
        log.info("Scanning for relevant JS/TS files...")
        for file in self.repo_path.rglob("*"):
            if file.suffix in {".js", ".ts", ".tsx", ".jsx", ".mjs"} and self._is_valid_file(file):
                log.debug("Found relevant JS/TS file", file=str(file))
                yield file

    def get_network_related_files(self, files: List[Path]) -> Generator[Path, None, None]:
        log.info("Scanning for network-related JS/TS files...")
        for file in files:
            try:
                if self._contains_patterns(file, self.compiled_patterns):
                    log.debug("Network-related JS/TS file detected", file=str(file))
                    yield file
            except UnicodeDecodeError:
                log.warning("Skipping non-text file", file=str(file))
                continue
            except Exception as e:
                # log.error("Error processing file", file=str(file), error=str(e))
                continue

    def _is_valid_file(self, file: Path) -> bool:
        file_str = str(file).replace('\\', '/').lower()
        return not any(exclusion in file_str for exclusion in self.to_exclude) and not any(fn in file.name.lower() for fn in self.file_names_to_exclude)

    def _contains_patterns(self, file: Path, patterns: List[re.Pattern]) -> bool:
        try:
            content = file.read_text(encoding='utf-8', errors='ignore')
            return any(pattern.search(content) for pattern in patterns)
        except Exception as e:
            log.error("Error reading file", file=str(file), error=str(e))
            return False

    def analyze_repository(self) -> dict:
        log.info("Starting JS/TS repository analysis")
        js_ts_files = list(self.get_relevant_files())
        js_ts_network_files = list(self.get_network_related_files(js_ts_files))

        log.info("Analysis complete", total_js_ts_files=len(js_ts_files), network_js_ts_files=len(js_ts_network_files))
        return {
            "total_js_ts_files": len(js_ts_files),
            "network_js_ts_files": len(js_ts_network_files),
            "js_ts_network_files_list": [str(f.relative_to(self.repo_path)) for f in js_ts_network_files]
        }
