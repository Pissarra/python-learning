# Python Learning

Projeto de aprendizado de Python com foco em Machine Learning e Deep Learning.

## Pré-requisitos

- Python >= 3.13

## Instalação

### 1. Instalar dependências do sistema

Para usar o módulo de reconhecimento de voz (`speech_recognition_example.py`), é necessário instalar o PortAudio:

```bash
brew install portaudio
```

### 2. Instalar dependências Python

O projeto usa `uv` para gerenciar dependências. Para instalar todas as dependências:

```bash
uv sync
```

Isso criará um ambiente virtual em `.venv` e instalará todas as dependências listadas no `pyproject.toml`.

## Bibliotecas Principais

As seguintes bibliotecas são instaladas automaticamente ao executar `uv sync`:

- **matplotlib** (>=3.10.5) - Visualização de dados
- **pandas** (>=2.3.1) - Manipulação e análise de dados
- **pandas-stubs** (==2.3.0.250703) - Type stubs para pandas
- **plotly** (>=6.3.0) - Gráficos interativos
- **scikit-learn** (>=1.7.1) - Machine Learning
- **SpeechRecognition** (>=3.10.0) - Reconhecimento de voz
- **pyaudio** (>=0.2.14) - Interface para áudio (requer PortAudio instalado)

## Estrutura do Projeto

```
python-learning/
├── deep-learning/          # Scripts de Deep Learning
├── machine-learning/       # Scripts de Machine Learning
├── view-chart/             # Visualizações
└── pyproject.toml          # Configuração do projeto e dependências
```

## Notas


- Para scripts que usam microfone (speech_recognition), certifique-se de que o PortAudio está instalado
- Alguns scripts podem requerer permissões de acesso ao microfone (macOS)
