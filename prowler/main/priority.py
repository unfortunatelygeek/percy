import os
import re
import json
from pathlib import Path
from typing import List, Set

def get_modified(changelog_path: str) -> Set[str]:
    mods = set()
    pattern = re.compile(r'\*\s+([a-zA-Z0-9_/]+)')
    with open(changelog_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                mods.add(match.group(1))
    return mods

def get_tests(repo_path: str, mods: Set[str]) -> List[str]:
    tests = []
    for root, _, files in os.walk(repo_path):
        if "tests" in root.lower() or "unittest" in root.lower():
            for file in files:
                if file.endswith(".py"):
                    path = os.path.relpath(os.path.join(root, file), repo_path)
                    if any(comp in path for comp in mods):
                        tests.append(path)
    return tests

def main():
    repo_path = os.getenv("REPO_PATH", ".")
    changelog_path = os.path.join(repo_path, "CHANGELOG.md")
    if not os.path.exists(changelog_path):
        print("Changelog not found. Running all tests.")
        return
    mods = get_modified(changelog_path)
    tests = get_tests(repo_path, mods)
    print(json.dumps({"tests_to_run": tests}, indent=4))

if __name__ == "__main__":
    main()

