import networkx as nx
from tree_sitter_language_pack import get_parser
from .vuln_extract import VulnerabilityExtractor

def traverse_ast(node, vulnerabilities):
    """Recursively traverses the AST and checks for vulnerabilities."""
    if not node:
        return

    found_vulns = VulnerabilityExtractor.check_vulnerability(node)
    if found_vulns:
        vulnerabilities.extend(found_vulns)

    for child in node.children:
        traverse_ast(child, vulnerabilities)

def analyze_code(code, language='javascript'):
    """Parses code into AST and analyzes vulnerabilities."""
    parser = get_parser(language)
    tree = parser.parse(code.encode("utf-8"))
    G = nx.DiGraph()

    vulnerabilities_found = []
    traverse_ast(tree.root_node, vulnerabilities_found)

    if vulnerabilities_found:
        print("\n=== Detected Vulnerabilities ===")
        for vuln in vulnerabilities_found:
            print(f"🔴 {vuln['type']}: {vuln['description']}\n   Code: {vuln['code']}\n")

    return G, vulnerabilities_found
