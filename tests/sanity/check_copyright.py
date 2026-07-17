# Copyright 2026 RL-bMAS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path

copyright_notice = re.compile(r"^[ \t]*#.*\bcopyright\b", re.IGNORECASE | re.MULTILINE)


def tracked_python_files(directory: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={directory.as_posix()}",
            "-C",
            str(directory),
            "ls-files",
            "*.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [directory / relative_path for relative_path in result.stdout.splitlines()]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=Path)
    args = parser.parse_args()

    files = tracked_python_files(args.directory.resolve())
    missing = []
    for path in files:
        file_content = path.read_text(encoding="utf-8")
        if copyright_notice.search(file_content) is None:
            missing.append(path)

    assert not missing, "files do not contain a copyright notice:\n" + "\n".join(map(str, missing))
    print(f"Copyright check passed for {len(files)} tracked Python files.")
