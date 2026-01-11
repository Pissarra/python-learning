# Compatibility module for google.colab
import sys
from pathlib import Path

# Garante que o diretório raiz está no path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
