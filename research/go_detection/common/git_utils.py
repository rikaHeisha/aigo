import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

THIS_FOLDER = os.path.dirname(__file__)


@dataclass
class GitInfo:
    current_commit: str
    current_commit_short: str
    branch_name: str
    repo_clean: bool


def _fetch_git_properties(base_path: str) -> GitInfo:
    assert os.path.isdir(
        os.path.join(base_path, ".git")
    ), f"Did not find .git folder in: {base_path}"
    # print(f"Found git directory at: {base_path}")

    # return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    def _run(cmd):
        result = (
            subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                cwd=base_path,
            )
            .decode("ascii")
            .strip()
        )
        return result

    # Properties
    current_commit = _run(["git", "rev-parse", "HEAD"])
    current_commit_short = _run(["git", "rev-parse", "--short", "HEAD"])
    branch_name = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    repo_clean = _run(["git", "status", "--porcelain"]) == ""

    return GitInfo(
        current_commit,
        current_commit_short,
        branch_name,
        repo_clean,
    )


def get_git_info(
    base_path: Optional[str] = None, recurse_parents: bool = True
) -> GitInfo:

    git_base_path = None
    if not recurse_parents:
        git_base_path = base_path
    else:
        base_path = Path(base_path or THIS_FOLDER)
        parents = [base_path] + list(base_path.parents)

        for path in parents:
            path = str(path)
            if os.path.isdir(os.path.join(path, ".git")):
                # Found a git directory
                git_base_path = path
                break

    assert git_base_path is not None
    return _fetch_git_properties(git_base_path)
