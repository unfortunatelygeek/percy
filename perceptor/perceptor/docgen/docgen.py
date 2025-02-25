import os
import subprocess
import markdown
import ollama  # Replace with the LLM library you're using

def get_git_root():
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip()

def get_git_changes():
    log_output = subprocess.check_output(['git', 'log', '--patch', '-n', '5', '--pretty=format:"%h %s"']).decode()
    return log_output

def generate_documentation(snippet):
    prompt = f"""
    Document the following code changes in markdown:
    ```
    {snippet}
    ```
    """
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def update_markdown(file_path, new_content):
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write('\n' + new_content)
    else:
        with open(file_path, 'w') as f:
            f.write(new_content)

def main():
    repo_root = get_git_root()
    changes = get_git_changes()
    doc_text = generate_documentation(changes)
    
    current_dir = os.getcwd()
    readme_path = os.path.join(current_dir, 'README.md')
    update_markdown(readme_path, doc_text)
    
    changelog_path = os.path.join(repo_root, 'CHANGELOG.md')
    update_markdown(changelog_path, f'## Recent Changes\n{doc_text}')
    
    print(f"Updated {readme_path} and {changelog_path}")

if __name__ == "__main__":
    main()

