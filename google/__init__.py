# Compatibility module for google.colab
import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao PYTHONPATH
# O módulo google está em: projeto/google/
# Então o diretório raiz é o parent do diretório onde este arquivo está
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent  # google/ -> projeto/

# Adiciona o diretório raiz ao sys.path se não estiver lá
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    # Insere no início para ter prioridade sobre outros pacotes 'google'
    sys.path.insert(0, _project_root_str)
