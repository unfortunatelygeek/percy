# import networkx as nx
# from tree_sitter_language_pack import get_parser
# from .vuln_extract import VulnerabilityExtractor

# def traverse_ast(node, vulnerabilities):
#     """Recursively traverses the AST and checks for vulnerabilities."""
#     if not node:
#         return

#     found_vulns = VulnerabilityExtractor.check_vulnerability(node)
#     if found_vulns:
#         vulnerabilities.extend(found_vulns)

#     for child in node.children:
#         traverse_ast(child, vulnerabilities)

# def analyze_code(code, language='javascript'):
#     """Parses code into AST and analyzes vulnerabilities."""
#     parser = get_parser(language)
#     tree = parser.parse(code.encode("utf-8"))
#     G = nx.DiGraph()

#     vulnerabilities_found = []
#     traverse_ast(tree.root_node, vulnerabilities_found)

#     if vulnerabilities_found:
#         print("\n=== Detected Vulnerabilities ===")
#         for vuln in vulnerabilities_found:
#             print(f"🔴 {vuln['type']}: {vuln['description']}\n   Code: {vuln['code']}\n")

#     return G, vulnerabilities_found

import networkx as nx
from tree_sitter_language_pack import get_parser
from .vuln_extract import VulnerabilityExtractor

def analyze_code(code, language='javascript'):
    """
    Parses code into AST and analyzes vulnerabilities.
    
    Args:
        code (str): The source code to analyze
        language (str): Programming language of the code
        
    Returns:
        tuple: (nx.DiGraph, list) - Graph representation and vulnerabilities found
    """
    # Get appropriate parser from tree-sitter
    parser = get_parser(language)
    
    # Parse code into AST
    tree = parser.parse(code.encode("utf-8"))
    
    # Create a graph representation
    G = nx.DiGraph()

    # Find vulnerabilities
    vulnerabilities_found = []
    traverse_ast(tree.root_node, vulnerabilities_found)

    # Print found vulnerabilities
    if vulnerabilities_found:
        print("\n=== Detected Vulnerabilities ===")
        for vuln in vulnerabilities_found:
            print(f"🔴 {vuln['type']}: {vuln['description']}\n   Code: {vuln['code']}\n")

    return G, vulnerabilities_found

def traverse_ast(node, vulnerabilities):
    """
    Recursively traverses the AST and checks for vulnerabilities.
    
    Args:
        node: The current AST node
        vulnerabilities (list): List to collect found vulnerabilities
    """
    if not node:
        return

    # Check for vulnerabilities in current node
    found_vulns = VulnerabilityExtractor.check_vulnerability(node)
    if found_vulns:
        vulnerabilities.extend(found_vulns)

    # Recursively check children
    for child in node.children:
        traverse_ast(child, vulnerabilities)