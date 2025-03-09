import subprocess
from pathlib import Path
from typing import List
from prowler.core.logger import log

class GitDiffAnalyzer:
    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path).resolve()
        log.info("GitDiffAnalyzer initialized", repo_path=str(self.repo_path))

    def get_changed_files(self) -> List[Path]:
        """Fetches the list of modified files from the latest Git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.repo_path,
                text=True,
                capture_output=True,
                check=True
            )
            changed_files = [self.repo_path / Path(file) for file in result.stdout.strip().split("\n") if file]
            log.info("Changed files detected", count=len(changed_files))
            return changed_files
        except subprocess.CalledProcessError as e:
            log.error("Error fetching Git diff", error=str(e))
            return []

    def analyze_git_diff(self) -> dict:
        """Analyzes only the files changed in the latest commit."""
        changed_files = self.get_changed_files()
        js_ts_files = [f for f in changed_files if f.suffix in {".js", ".ts", ".tsx", ".jsx", ".mjs"}]

        log.info("Git diff analysis complete", total_changed_files=len(js_ts_files))
        return {
            "total_changed_files": len(js_ts_files),
            "changed_files_list": [str(f.relative_to(self.repo_path)) for f in js_ts_files]
        }
