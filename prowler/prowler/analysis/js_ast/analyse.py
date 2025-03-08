from .parser import analyze_code
from .visualiser import visualize_vulnerabilities

def run_analysis(code, language='javascript', max_depth=8):
    """Runs vulnerability analysis and visualizes the AST."""
    G, vulnerabilities = analyze_code(code, language, max_depth)
    
    print("\nVULNERABILITIES DETECTED:")
    for i, vuln in enumerate(vulnerabilities, 1):
        print(f"{i}. In {vuln['node_label']}:")
        for issue in vuln['issues']:
            print(f"   - {issue[0]} (Pattern: {issue[1]})")
    
    visualize_vulnerabilities(G)
