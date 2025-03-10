import jedi
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
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
    print("JS was loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Tree-sitter JS/TS language: {e}")
    JS_PARSER = None

# class SymbolExtractor:
#     def __init__(self, files_to_analyze: List[Path]) -> None:
#         self.files_to_analyze = files_to_analyze
#         self.project = jedi.Project(files_to_analyze[0].parent if files_to_analyze else Path.cwd())
    
#     def extract(self, symbol_name: str) -> Optional[Dict[str, Any]]:
#         if not self.files_to_analyze:
#             print("No files to analyze.")
#             return None
        
#         if self.files_to_analyze[0].suffix == ".py":
#             return self._extract_python_symbol(symbol_name)
#         else:
#             return self._extract_js_ts_symbol(symbol_name)
    
#     def _extract_python_symbol(self, symbol_name: str) -> Optional[Dict[str, Any]]:
#         scripts = [jedi.Script(path=str(file), project=self.project) for file in self.files_to_analyze]
#         return self._search_for_symbol(symbol_name, scripts)
    
#     def _search_for_symbol(self, symbol_name: str, scripts) -> Optional[Dict[str, Any]]:
#         for script in scripts:
#             completions = script.complete(line=1, column=0)
#             for completion in completions:
#                 if completion.name == symbol_name:
#                     return {
#                         "name": symbol_name,
#                         "file_path": completion.module_path,
#                         "source": completion.get_line_code(),
#                         "type": "python symbol"
#                     }
#         return None
    
#     def _extract_js_ts_symbol(self, symbol_name: str) -> Optional[Dict[str, Any]]:
#         if JS_PARSER is None:
#             print("Error: Tree-sitter JS/TS parser is not initialized.")
#             return None

#         for file in self.files_to_analyze:
#             try:
#                 with file.open(encoding='utf-8') as f:
#                     content = f.read()
#                 tree = JS_PARSER.parse(content.encode('utf-8'))
#                 match = self._search_tree_for_symbol(symbol_name, tree, content, str(file))
#                 if match:
#                     return match
#             except Exception as e:
#                 print(f"Error processing {file}: {e}")
#         return None
    
#     def _search_tree_for_symbol(self, symbol_name: str, tree, content: str, file_path: str) -> Optional[Dict[str, Any]]:
#         def traverse(node):
#             if node.type in {"function_declaration", "class_declaration", "variable_declarator", "arrow_function", "method_definition"}:
#                 for child in node.children:
#                     if child.type == "identifier" and content[child.start_byte:child.end_byte] == symbol_name:
#                         return {
#                             "name": symbol_name,
#                             "file_path": file_path,
#                             "source": content[node.start_byte:node.end_byte].strip(),
#                             "type": node.type.replace("_", " ")
#                         }
#             for child in node.children:
#                 match = traverse(child)
#                 if match:
#                     return match
#             return None
        
#         return traverse(tree.root_node)

# import jedi
# import json
# from pathlib import Path
# from typing import List, Dict, Any, Union, Optional
# from tree_sitter import Language, Parser

# # Initialize Tree-sitter for JS/TS
# JS_PARSER = None
# try:
#     JS_PARSER = Parser()
#     language_dir = Path('../../tree-sitter-javascript')
    
#     if language_dir.exists():
#         Language.build_library(
#             'build/my-languages.so',
#             [str(language_dir), str(Path('tree-sitter-python'))]
#         )
#         JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
#         JS_PARSER.set_language(JS_LANGUAGE)
#         print("JS parser loaded successfully.")
#     else:
#         print("Warning: JS language directory not found")
        
# except Exception as e:
#     print(f"Warning: Could not load Tree-sitter JS/TS language: {e}")
#     JS_PARSER = None

class SymbolExtractor:
    """
    Extracts symbols from Python, JavaScript, or TypeScript files.
    """
    
    def __init__(self, root_dir: str = None, files_to_analyze: List[Path] = None) -> None:
        """
        Initialize the Symbol Extractor.
        
        Args:
            root_dir (str): Project root directory
            files_to_analyze (List[Path]): List of files to analyze
        """
        self.files_to_analyze = files_to_analyze or []
        self.project = None
        
        if self.files_to_analyze:
            project_path = Path(root_dir) if root_dir else self.files_to_analyze[0].parent
            self.project = jedi.Project(project_path)
    
    def extract(self, symbol_name: str, files_to_analyze: List[Path] = None) -> List[Dict[str, Any]]:
        """
        Extract symbols matching the given name from files.
        
        Args:
            symbol_name (str): Symbol name to search for
            files_to_analyze (List[Path]): Optional override of files to analyze
            
        Returns:
            List[Dict[str, Any]]: List of extracted symbols
        """
        if files_to_analyze:
            self.files_to_analyze = files_to_analyze
            
        if not self.files_to_analyze:
            print("No files to analyze.")
            return []
        
        results = []
        for file in self.files_to_analyze:
            if str(file).endswith(('.py')):
                symbol = self._extract_python_symbol(symbol_name, file)
            else:
                symbol = self._extract_js_ts_symbol(symbol_name, file)
                
            if symbol:
                results.append(symbol)
                
        return results
    
    def _extract_python_symbol(self, symbol_name: str, file: Path) -> Optional[Dict[str, Any]]:
        """Extract symbol from Python file using Jedi."""
        try:
            script = jedi.Script(path=str(file), project=self.project)
            completions = script.complete(line=1, column=0)
            
            for completion in completions:
                if completion.name == symbol_name:
                    return {
                        "name": symbol_name,
                        "file_path": str(completion.module_path),
                        "source": completion.get_line_code(),
                        "type": "python symbol"
                    }
        except Exception as e:
            print(f"Error extracting Python symbol from {file}: {e}")
            
        return None
    
    def _extract_js_ts_symbol(self, symbol_name: str, file: Path) -> Optional[Dict[str, Any]]:
        """Extract symbol from JavaScript/TypeScript file using Tree-sitter."""
        if JS_PARSER is None:
            print("Error: Tree-sitter JS/TS parser is not initialized.")
            return None

        try:
            with open(file, encoding='utf-8') as f:
                content = f.read()
                
            tree = JS_PARSER.parse(content.encode('utf-8'))
            return self._search_tree_for_symbol(symbol_name, tree, content, str(file))
            
        except Exception as e:
            print(f"Processing {file}: {e}")            
        return None
    
    def _search_tree_for_symbol(self, symbol_name: str, tree, content: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Search AST tree for symbols matching the given name."""
        def traverse(node):
            # Check if current node might contain our symbol
            if node.type in {"function_declaration", "class_declaration", "variable_declarator", 
                            "arrow_function", "method_definition"}:
                for child in node.children:
                    if child.type == "identifier" and content[child.start_byte:child.end_byte] == symbol_name:
                        return {
                            "name": symbol_name,
                            "file_path": file_path,
                            "source": content[node.start_byte:node.end_byte].strip(),
                            "type": node.type.replace("_", " ")
                        }
            
            # Recursively check all children
            for child in node.children:
                match = traverse(child)
                if match:
                    return match
                    
            return None
        
        return traverse(tree.root_node)