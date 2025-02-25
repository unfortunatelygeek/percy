import jedi
from pathlib import Path
from typing import List, Dict, Any

class SymbolExtractor:
    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)
        self.project = jedi.Project(self.repo_path)
        self.ignore = ['/test', '_test/', '/docs', '/example']

    # extracts the definition of a symbol from the repository using Jedi
    def extract(self, symbol_name: str, code_line: str, filtered_files: List[Path]) -> Dict[str, Any] | None:
        symbol_parts = symbol_name.split('.')
        matching_files = [file for file in filtered_files if self._search_string_in_file(file, code_line)]

        if not matching_files:
            return None

        scripts = [jedi.Script(path=str(file), project=self.project) for file in matching_files]

        match = self._search_for_symbol(symbol_name, scripts)
        return match

    # uses jedi to find symbol definitions
    def _search_for_symbol(self, symbol_name: str, scripts: List[jedi.Script]) -> Dict[str, Any] | None:
        for script in scripts:
            results = script.search(symbol_name)
            for name in results:
                if name.type in ['function', 'class', 'statement', 'instance', 'module']:
                    return self._create_match_obj(name, symbol_name)
        return None

    # Checks if a string is present in a given file
    def _search_string_in_file(self, file_path: Path, string: str) -> bool:
        with file_path.open(encoding='utf-8') as file:
            content = file.read().replace(' ', '').replace('\n', '')
            return string.replace(' ', '').replace('\n', '') in content

    # Formats the extracted match into a dictionary
    def _create_match_obj(self, name, symbol_name: str) -> Dict[str, Any]:
        module_path = str(name.module_path)
        start, end = name.get_definition_start_position(), name.get_definition_end_position()

        with Path(module_path).open(encoding='utf-8') as f:
            lines = f.readlines()
            source = ''.join(lines[start[0]-1:end[0]]) if start and end else ''.join(lines)

        return {
            "name": name.name,
            "context_name_requested": symbol_name,
            "file_path": module_path,
            "source": source.strip()
        }
