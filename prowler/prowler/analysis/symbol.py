import jedi
import tree_sitter
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Generator
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


class SymbolExtractor:
    """
    Extracts symbol definitions from Python and JavaScript/TypeScript files.
    Uses Jedi for Python and Tree-sitter for JS/TS.
    """
    
    def __init__(self, repo_path: Union[str, Path]) -> None:
        self.repo_path = Path(repo_path)
        self.project = jedi.Project(self.repo_path)
        self.ignore_paths = {'/test', '_test/', '/docs', '/example', '/node_modules', '/__pycache__', '/dist'}
    
    def get_filtered_files(self, extensions: List[str]) -> Generator[Path, None, None]:
        """Get files with specified extensions, excluding paths in ignore_paths."""
        for ext in extensions:
            for file in self.repo_path.rglob(f"*{ext}"):
                file_str = str(file).replace('\\', '/')
                if not any(ignore in file_str for ignore in self.ignore_paths):
                    yield file
    
    def extract(self, symbol_name: str, code_line: str, filtered_files: List[Path]) -> Optional[Dict[str, Any]]:
        """
        Extracts the definition of a symbol from Python or JS/TS files.
        """
        matching_files = self._find_matching_files(code_line, filtered_files)
        
        if not matching_files:
            return None
        
        if matching_files[0].suffix == ".py":
            return self._extract_python_symbol(symbol_name, matching_files)
        else:  # JS/TS files
            return self._extract_js_ts_symbol(symbol_name, matching_files)
    
    def extract_by_file_type(self, symbol_name: str, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract a symbol directly from a specific file."""
        if file_path.suffix == ".py":
            return self._extract_python_symbol(symbol_name, [file_path])
        else:  # JS/TS files
            return self._extract_js_ts_symbol(symbol_name, [file_path])
    
    def _find_matching_files(self, code_line: str, filtered_files: List[Path]) -> List[Path]:
        """Find files that might contain the specified code line."""
        # Implement your matching logic here
        # For simplicity, just return all files for this example
        return filtered_files
    
    def _extract_python_symbol(self, symbol_name: str, matching_files: List[Path]) -> Optional[Dict[str, Any]]:
        """Extracts a symbol using Jedi for Python files."""
        scripts = [jedi.Script(path=str(file), project=self.project) for file in matching_files]
        return self._search_for_symbol(symbol_name, scripts)
    
    def _search_for_symbol(self, symbol_name: str, scripts) -> Optional[Dict[str, Any]]:
        """Search for a symbol in Jedi scripts."""
        # Implement your Jedi search logic here
        # This is a placeholder
        for script in scripts:
            completions = script.complete(line=1, column=0)
            for completion in completions:
                if completion.name == symbol_name:
                    return {
                        "name": symbol_name,
                        "context_name_requested": symbol_name,
                        "file_path": completion.module_path,
                        "source": completion.get_line_code(),
                        "type": "python symbol"
                    }
        return None
    
    def _extract_js_ts_symbol(self, symbol_name: str, matching_files: List[Path]) -> Optional[Dict[str, Any]]:
        """Extracts a symbol using Tree-sitter for JS/TS files."""
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
        """
        Traverses the Tree-sitter AST to find symbol definitions.
        Supports:
        - Function declarations
        - Class declarations
        - Variable declarations
        - Arrow functions
        - Method definitions
        """
        def traverse(node):
            if node.type in {"function_declaration", "class_declaration", "variable_declarator", "arrow_function", "method_definition"}:
                for child in node.children:
                    if child.type == "identifier" and content[child.start_byte:child.end_byte] == symbol_name:
                        return {
                            "name": symbol_name,
                            "context_name_requested": symbol_name,
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