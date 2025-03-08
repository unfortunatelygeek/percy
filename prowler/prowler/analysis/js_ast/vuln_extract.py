class VulnerabilityExtractor:
    """Detects vulnerabilities in source code by analyzing AST nodes."""

    VULNERABILITY_PATTERNS = {
        "LFI": {
            "type": "call_expression",
            "name": "open",
            "description": "Local File Inclusion (LFI) - Unvalidated file access",
            "argument_check": ["user", "input", "request", "param"]
        },
        "AFO": {
            "type": "call_expression",
            "name": "write",
            "description": "Arbitrary File Overwrite (AFO) - Unvalidated file write",
            "argument_check": ["user", "input", "request", "param"]
        },
        "RCE": {
            "type": "call_expression",
            "name": "eval",
            "description": "Remote Code Execution (RCE) - Unvalidated eval execution",
            "argument_check": ["user", "input", "request", "param"]
        },
        "XSS": {
            "type": "assignment_expression",
            "property": "innerHTML",
            "description": "Cross-Site Scripting (XSS) - Unvalidated innerHTML assignment"
        },
        "SQLI": {
            "type": "binary_expression",
            "contains": "SELECT",
            "description": "SQL Injection (SQLI) - String concatenation in SQL query"
        },
        "SSRF": {
            "type": "call_expression",
            "name": ["requests.get", "urllib.request"],
            "description": "Server-Side Request Forgery (SSRF) - Unvalidated external request",
            "argument_check": ["user", "input", "request", "param"]
        },
        "IDOR": {
            "type": "binary_expression",
            "contains": "id",
            "description": "Insecure Direct Object Reference (IDOR) - Unrestricted object access"
        }
    }

    @staticmethod
    def check_vulnerability(node):
        """Checks if an AST node contains known vulnerabilities"""
        vulnerabilities = []
        node_text = node.text.decode('utf-8') if node.text else ""

        for vuln_name, pattern in VulnerabilityExtractor.VULNERABILITY_PATTERNS.items():
            # Check call expressions (e.g., eval, open, write)
            if pattern["type"] == "call_expression" and node.type == "call_expression":
                function_node = node.child_by_field_name("function")
                if function_node and function_node.text:
                    function_name = function_node.text.decode("utf-8")
                    if function_name in (pattern["name"] if isinstance(pattern["name"], list) else [pattern["name"]]):
                        # Check arguments for user input sources
                        for arg in pattern.get("argument_check", []):
                            if arg in node_text:
                                vulnerabilities.append({
                                    "type": vuln_name,
                                    "description": pattern["description"],
                                    "code": node_text
                                })
            
            # Check assignment expressions (e.g., innerHTML)
            if pattern["type"] == "assignment_expression" and node.type == "assignment_expression":
                left_node = node.child_by_field_name("left")
                if left_node and left_node.text and pattern["property"] in left_node.text.decode("utf-8"):
                    vulnerabilities.append({
                        "type": vuln_name,
                        "description": pattern["description"],
                        "code": node_text
                    })
            
            # Check binary expressions (e.g., SQL Injection, IDOR)
            if pattern["type"] == "binary_expression" and node.type == "binary_expression":
                if pattern["contains"] in node_text:
                    vulnerabilities.append({
                        "type": vuln_name,
                        "description": pattern["description"],
                        "code": node_text
                    })

        return vulnerabilities
