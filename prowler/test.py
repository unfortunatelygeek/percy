import networkx as nx
import matplotlib.pyplot as plt
from tree_sitter_language_pack import get_parser

# Enhanced vulnerability detection patterns
VULNERABILITY_PATTERNS = {
    "eval_usage": {
        "type": "call_expression",
        "name": "eval",
        "description": "Insecure use of eval() - Code Injection Risk"
    },
    "document_write": {
        "type": "call_expression",
        "name": "document.write",
        "description": "Unsafe document.write() - XSS Risk"
    },
    "innerHTML": {
        "type": "assignment_expression",
        "property": "innerHTML",
        "description": "Unvalidated innerHTML assignment - XSS Risk"
    },
    "sql_string_concat": {
        "type": "binary_expression",
        "contains": "SELECT",
        "description": "String concatenation in SQL query - SQL Injection Risk"
    },
    "password_plaintext": {
        "type": "property_identifier",
        "contains": "password",
        "description": "Plaintext password handling - Security Risk"
    }
}

def check_vulnerability(node):
    """Check if a node contains any known vulnerabilities"""
    vulnerabilities = []
    node_text = node.text.decode('utf-8') if node.text else ""
    
    # Check for eval usage
    if (node.type == "call_expression" and 
        node.child_by_field_name("function") and 
        node.child_by_field_name("function").text and
        node.child_by_field_name("function").text.decode('utf-8') == "eval"):
        vulnerabilities.append(VULNERABILITY_PATTERNS["eval_usage"]["description"])
    
    # Check for document.write usage
    if (node.type == "call_expression" and 
        node.child_by_field_name("function") and 
        node.child_by_field_name("function").text and
        "document.write" in node.child_by_field_name("function").text.decode('utf-8')):
        vulnerabilities.append(VULNERABILITY_PATTERNS["document_write"]["description"])
    
    # Check for innerHTML assignment
    if (node.type == "assignment_expression" and 
        node.child_by_field_name("left") and 
        node.child_by_field_name("left").text and
        "innerHTML" in node.child_by_field_name("left").text.decode('utf-8')):
        vulnerabilities.append(VULNERABILITY_PATTERNS["innerHTML"]["description"])
    
    # Check for SQL injection
    if (node.type == "binary_expression" and 
        node.child_by_field_name("right") and 
        node.child_by_field_name("right").type == "string_literal" and
        "SELECT" in node_text):
        vulnerabilities.append(VULNERABILITY_PATTERNS["sql_string_concat"]["description"])
    
    # Check for password plaintext
    if "password" in node_text.lower() and node.type == "property_identifier":
        vulnerabilities.append(VULNERABILITY_PATTERNS["password_plaintext"]["description"])
    
    return vulnerabilities

def build_simplified_ast(node, G, parent=None, depth=0, max_depth=None):
    """Build a simplified AST graph, focusing only on meaningful nodes"""
    # Skip nodes that add visual noise
    if node.type in ['(', ')', '[', ']', '{', '}', '"', "'", ";", ","]:
        return
    
    # Stop at max depth if specified
    if max_depth is not None and depth > max_depth:
        return
    
    # Check if node is significant
    is_significant = (
        node.type in ["function_declaration", "call_expression", "variable_declaration", 
                     "assignment_expression", "binary_expression", "if_statement", 
                     "return_statement", "identifier", "property_identifier"]
    )
    
    # Only add significant nodes or nodes with vulnerabilities
    vulnerabilities = check_vulnerability(node)
    if is_significant or vulnerabilities:
        # Create simplified label
        if node.type == "call_expression" and node.child_by_field_name("function"):
            label = f"{node.child_by_field_name('function').text.decode('utf-8')}()"
        elif node.type == "function_declaration" and node.child_by_field_name("name"):
            label = f"function {node.child_by_field_name('name').text.decode('utf-8')}()"
        elif node.type == "variable_declaration":
            label = "variable declaration"
        else:
            label = node.type
            if node.text and len(node.text) < 30:  # Limit text length to prevent huge nodes
                text = node.text.decode('utf-8')
                if not text.isspace() and text not in ["{", "}"]:
                    label = f"{node.type}: {text}"
        
        node_id = id(node)
        G.add_node(node_id, label=label, vulnerabilities=vulnerabilities, node_type=node.type)
        
        if parent:
            G.add_edge(parent, node_id)
        
        parent = node_id
    
    # Process children
    for child in node.children:
        build_simplified_ast(child, G, parent, depth + 1, max_depth)

def analyze_code(code, language='javascript', max_depth=None):
    """Analyze code for vulnerabilities and generate visualization"""
    # Load parser for the specified language
    parser = get_parser(language)
    
    # Parse the code into an AST
    tree = parser.parse(code.encode("utf-8"))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Build simplified AST
    build_simplified_ast(tree.root_node, G, max_depth=max_depth)
    
    # Get vulnerability summary
    vulnerabilities_found = []
    for node in G.nodes:
        if G.nodes[node]['vulnerabilities']:
            vuln_info = {
                'node_label': G.nodes[node]['label'],
                'issues': G.nodes[node]['vulnerabilities']
            }
            vulnerabilities_found.append(vuln_info)
    
    return G, vulnerabilities_found

def hierarchical_layout(G):
    """Create a hierarchical layout without using pygraphviz"""
    # Set up initial positions by layer
    pos = {}
    layers = {}
    
    # Identify root nodes (nodes with no incoming edges)
    root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    
    # If no root nodes found, pick any node as root
    if not root_nodes:
        root_nodes = [list(G.nodes())[0]] if G.nodes() else []
    
    # BFS to assign layers
    current_layer = 0
    next_layer = set(root_nodes)
    
    while next_layer:
        current_nodes = next_layer
        layers[current_layer] = list(current_nodes)
        next_layer = set()
        
        for node in current_nodes:
            for successor in G.successors(node):
                next_layer.add(successor)
        
        current_layer += 1
    
    # Position nodes within layers
    max_nodes_per_layer = max((len(nodes) for nodes in layers.values()), default=0)
    layer_width = max(1.0, max_nodes_per_layer * 1.5)  # Scale width based on most populated layer
    
    for layer_num, nodes in layers.items():
        y_position = -layer_num * 2.0  # Vertical spacing between layers
        
        if not nodes:
            continue
        
        # Distribute nodes horizontally within the layer
        if len(nodes) == 1:
            x_position = 0  # Center single node
        else:
            spacing = layer_width / (len(nodes) - 1) if len(nodes) > 1 else 1
            for i, node in enumerate(nodes):
                pos[node] = (-layer_width/2 + i * spacing, y_position)
    
    # Handle any nodes not assigned positions (unlikely but possible with cycles)
    for node in G.nodes():
        if node not in pos:
            # Find any positioned neighbor
            for neighbor in list(G.predecessors(node)) + list(G.successors(node)):
                if neighbor in pos:
                    # Position relative to neighbor
                    pos[node] = (pos[neighbor][0] + 0.5, pos[neighbor][1] - 1.0)
                    break
            else:
                # If no positioned neighbors, assign default position
                pos[node] = (0, -current_layer)
                current_layer += 1
    
    return pos

def visualize_vulnerabilities(G, output_file=None):
    """Visualize the AST with vulnerabilities highlighted"""
    plt.figure(figsize=(12, 10))
    
    # Use custom hierarchical layout
    pos = hierarchical_layout(G)
    
    # Normal nodes
    normal_nodes = [n for n in G.nodes if not G.nodes[n]['vulnerabilities']]
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=normal_nodes,
                          node_color='lightblue', 
                          node_size=1000,
                          alpha=0.8)
    
    # Vulnerable nodes
    vuln_nodes = [n for n in G.nodes if G.nodes[n]['vulnerabilities']]
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=vuln_nodes,
                          node_color='red', 
                          node_size=1200,
                          alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, arrows=True)
    
    # Draw labels with smaller font size
    labels = {node: G.nodes[node]['label'] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.axis('off')
    plt.title("Code Vulnerability Analysis")
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Sample JavaScript code with vulnerabilities
    js_code = """
    function greet(name) {
        eval(name); // Insecure use of eval()
        document.write("<p>Hello, " + name + "!</p>"); // Unsafe document.write
        
        var element = document.getElementById('greeting');
        element.innerHTML = "<p>Hello, " + name + "!</p>"; // Unsafe innerHTML
        
        var query = "SELECT * FROM users WHERE name = '" + name + "'"; // SQL injection
        
        var user = {
            username: "admin",
            password: "admin123" // Plaintext password
        };
        
        return "Greeting completed";
    }
    """
    
    # Analyze the code
    G, vulnerabilities = analyze_code(js_code, max_depth=8)
    
    # Display vulnerabilities found
    print("VULNERABILITIES DETECTED:")
    for i, vuln in enumerate(vulnerabilities, 1):
        print(f"{i}. In {vuln['node_label']}:")
        for issue in vuln['issues']:
            print(f"   - {issue}")
    
    # Visualize the code with vulnerabilities highlighted
    visualize_vulnerabilities(G)