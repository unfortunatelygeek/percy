import re
from pathlib import Path
from typing import List, Generator

class RepoAnalyzer:
    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path)
        self.to_exclude = {'/setup.py', '/test', '/example', '/docs', '/site-packages', '.venv', 'virtualenv', '/dist'}
        self.file_names_to_exclude = ['test_', 'conftest', '_test.py']

        self.patterns = [
            r'@app\.route\(.*?\)',  # Flask routes
            r'@router\.(?:get|post|put|delete|patch|options|head|trace)\(.*?\)',  # FastAPI
            r'@blueprint\.route\(.*?\)',  # Flask blueprints
            r'@websocket\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',  # WebSockets
            r'websockets\.serve\(.*?\)',
            r'async\sdef\s\w+\(.*?request',  # Async functions
        ]

        self.compiled_patterns = [re.compile(pattern) for pattern in self.patterns]

    def get_relevant_py_files(self) -> Generator[Path, None, None]:
        for f in self.repo_path.rglob("*.py"):
            f_str = str(f).replace('\\', '/').lower()

            if any(exclude in f_str for exclude in self.to_exclude):
                continue
            if any(fn in f.name for fn in self.file_names_to_exclude):
                continue

            yield f

    def get_network_related_files(self, files: List[Path]) -> Generator[Path, None, None]:
        for py_f in files:
            with py_f.open(encoding='utf-8') as f:
                content = f.read()
            if any(re.search(pattern, content) for pattern in self.compiled_patterns):
                yield py_f
