# main.py
from analysis.js_ast.parser import analyze_code

js_code = """
function processInput(userInput) {
    eval(userInput);
    document.write(userInput);
    var query = "SELECT * FROM users WHERE name = '" + userInput + "'";
    fetch(userInput);
}
"""

if __name__ == "__main__":
    analyze_code(js_code, language="javascript")
