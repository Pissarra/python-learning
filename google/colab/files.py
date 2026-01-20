"""
Compatibilidade para google.colab.files quando rodando localmente.
Simula o comportamento do files.upload() do Google Colab.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Dict

# Flag para controlar se já aplicamos o monkey patch
_patch_applied = False


def _setup_content_dir():
    """
    Cria o diretório /content/ para compatibilidade com código do Colab.
    Tenta criar link simbólico /content -> ./content, ou aplica monkey patch.
    """
    global _patch_applied
    
    # Cria ./content localmente
    local_content = Path('./content').resolve()
    local_content.mkdir(parents=True, exist_ok=True)
    
    # Tenta criar link simbólico /content -> ./content (pode falhar sem permissões)
    content_link = Path('/content')
    link_created = False
    
    if not content_link.exists():
        try:
            content_link.symlink_to(local_content)
            link_created = True
        except (OSError, PermissionError):
            pass
    elif content_link.is_symlink():
        try:
            if content_link.readlink() != local_content:
                content_link.unlink()
                content_link.symlink_to(local_content)
            link_created = True
        except (OSError, PermissionError):
            pass
    
    # Se não conseguiu criar o link, aplica monkey patch no keras.utils.load_img
    if not link_created and not _patch_applied:
        try:
            from keras.utils import load_img as _original_load_img
            
            def _patched_load_img(path, **kwargs):
                # Redireciona /content/ para ./content/
                if isinstance(path, str) and path.startswith('/content/'):
                    path = str(local_content / path[9:])  # Remove '/content/'
                return _original_load_img(path, **kwargs)
            
            # Aplica o monkey patch
            import keras.utils
            keras.utils.load_img = _patched_load_img
            _patch_applied = True
        except ImportError:
            # Keras ainda não foi importado, será aplicado quando necessário
            pass
    
    return local_content


class Files:
    """Classe que simula google.colab.files para uso local."""
    
    @staticmethod
    def upload() -> Dict[str, bytes]:
        """
        Simula files.upload() do Colab.
        Abre um diálogo para selecionar arquivos localmente.
        
        Returns:
            Dict[str, bytes]: Dicionário com nome do arquivo como chave e conteúdo como valor
        """
        # Configura diretório /content/ para compatibilidade
        content_dir = _setup_content_dir()
        
        root = tk.Tk()
        root.withdraw()  # Esconde a janela principal
        
        file_paths = filedialog.askopenfilenames(
            title="Selecione as imagens para fazer predição",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.gif *.bmp"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        root.destroy()
        
        uploaded = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    filename = os.path.basename(file_path)
                    # Salva em ./content/ (compatível com código que usa /content/)
                    dest_path = content_dir / filename
                    
                    with open(dest_path, 'wb') as dest:
                        content = f.read()
                        dest.write(content)
                        uploaded[filename] = content
                    
        return uploaded


def _apply_keras_patch():
    """Aplica monkey patch no keras.utils.load_img para redirecionar /content/"""
    global _patch_applied
    if _patch_applied:
        return
    
    try:
        # Tenta importar keras.utils
        import keras.utils
        from pathlib import Path
        
        local_content = Path('./content').resolve()
        local_content.mkdir(parents=True, exist_ok=True)
        
        if hasattr(keras.utils, 'load_img'):
            _original_load_img = keras.utils.load_img
            
            def _patched_load_img(path, **kwargs):
                # Redireciona /content/ para ./content/
                if isinstance(path, str) and path.startswith('/content/'):
                    path = str(local_content / path[9:])  # Remove '/content/'
                return _original_load_img(path, **kwargs)
            
            keras.utils.load_img = _patched_load_img
            _patch_applied = True
    except ImportError:
        # Keras ainda não está disponível, será aplicado quando necessário
        pass


# Aplica o patch quando o módulo é importado
_apply_keras_patch()

# Cria instância para compatibilidade com: from google.colab import files
files = Files()
