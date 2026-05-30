"""Put the repo root on sys.path so `import src` works under pytest.

Mirrors the Colab side, where the notebook does sys.path.insert(0, repo_root) after
git pull.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
