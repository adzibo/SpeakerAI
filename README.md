<div align="center">

```
███████╗██████╗ ███████╗ █████╗ ██╗  ██╗███████╗██████╗      █████╗ ██╗
██╔════╝██╔══██╗██╔════╝██╔══██╗██║ ██╔╝██╔════╝██╔══██╗    ██╔══██╗██║
███████╗██████╔╝█████╗  ███████║█████╔╝ █████╗  ██████╔╝    ███████║██║
╚════██║██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗ ██╔══╝  ██╔══██╗    ██╔══██║██║
███████║██║     ███████╗██║  ██║██║  ██╗███████╗██║  ██║    ██║  ██║██║
╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
```

**Transforma tu voz en un modelo de IA que habla por ti — completamente en local, con total privacidad y calidad de estudio.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Coqui TTS](https://img.shields.io/badge/Coqui_TTS-XTTS_v2-00B4D8?style=flat)](https://github.com/idiap/coqui-ai-TTS)
[![License](https://img.shields.io/badge/License-CPML-blueviolet?style=flat)](https://coqui.ai/cpml)
[![Created by](https://img.shields.io/badge/Created_by-AdZiBo-6C63FF?style=flat)](https://github.com/adzibo)

</div>

---

## ¿Qué es SpeakerAI?

SpeakerAI es una herramienta CLI de clonación de voz que corre íntegramente en tu máquina. Le das unos minutos de tu propia voz, entrena un modelo de IA y a partir de ese momento puede leer cualquier texto con ella — con tu timbre, tu cadencia, tu identidad sonora.

Sin servidores. Sin suscripciones. Sin que tu voz salga de tu equipo.

Construida sobre el modelo preentrenado **XTTS v2** de Coqui TTS, con fine-tuning del encoder GPT sobre datos propios y soporte para gestionar múltiples voces entrenadas de forma independiente.

---

## Demo

[![Demo SpeakerAI](https://img.youtube.com/vi/fwZ-aa9suf4/maxresdefault.jpg)](https://youtu.be/fwZ-aa9suf4)

> El vídeo muestra el flujo completo: clonación del repositorio, segmentación de audios con `splitter.py`, entrenamiento de dos voces distintas, generación de audio y muestra de resultados.

---

## Características

- **Clonación de voz en local** — todo el proceso ocurre en tu máquina, sin dependencias cloud
- **Fine-tuning sobre datos propios** — entrena el encoder GPT de XTTS v2 con tu propia voz
- **Gestión de múltiples voces** — entrena, lista y elimina voces de forma independiente
- **Validación automática del dataset** — verifica formato, sample rate y duraciones antes de entrenar
- **Speaker embedding cacheado** — inferencia rápida sin recalcular el embedding en cada generación
- **Segmentación automática** — `splitter.py` segmenta tus audios y genera las transcripciones
- **Soporte multilingüe** — español, inglés, francés, alemán, italiano, portugués y más
- **Monitorización con TensorBoard** — seguimiento de la curva de loss en tiempo real

---

## Entorno de desarrollo

| Componente | Mínimo recomendado |
|---|---|
| GPU | NVIDIA con 8 GB VRAM (testado en RTX 4080 SUPER 16 GB) |
| CUDA | 12.x |
| RAM | 32 GB |
| Almacenamiento | +100 GB libres |
| OS | Linux / WSL Ubuntu 22.04 |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/adzibo/SpeakerAI.git
cd SpeakerAI
```

### 2. Crear directorios necesarios:

```bash
mkdir datasets
mkdir models
mkdir models/base
mkdir models/trained
mkdir training
```

### 3. Crear y activar el entorno virtual

```bash
python3 -m venv ~/VenvTTS
source ~/VenvTTS/bin/activate
```

### 4. Instalar PyTorch con soporte CUDA

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

### 5. Instalar dependencias del proyecto

```bash
pip install -r requirements.txt
```

### 6. Descargar el modelo base XTTS v2

```bash
python3 speaker.py --download-base
```

> El modelo base (~2 GB) se descarga automáticamente desde Hugging Face y se copia a `models/base/`.

---

## Preparación del dataset

Los audios pasan por tres fases antes de estar listos para el entrenamiento:

**1. Audio Bruto** — graba tu voz con cualquier micrófono.

**2. Audio Tratado** — edita el audio en Audacity aplicando la siguiente cadena de efectos en orden:
- Efectos → Volumen y compresión → **Normalizar** (ajuste predeterminado vía LUFS)
- Efectos → Volumen y compresión → **Compresor** (preset: Podcast/Radio)
- Efectos → Volumen y compresión → **Limitador** (preset: SFX Limiter)
- Efectos → EQ y filtros → **Ecualizador** → Realce de graves
- Efectos → EQ y filtros → **Ecualizador** → Realce de agudos
- Efectos → EQ y filtros → **Ecualizador** → Atenuación progresiva para locuciones
- Efectos → Volumen y compresión → **Amplificar**

Exportar como **WAV · 22050 Hz · Mono · PCM 16-bit**.

**3. Audio de Entrenamiento** — segmenta el audio tratado con `splitter.py`:

```bash
python3 tools/splitter.py audioTratado.wav --name audioRaul
```

Esto genera la carpeta `datasets/audioRaul/` con la siguiente estructura:

```
datasets/
└── audioRaul/
    ├── wavs/
    │   ├── audioRaul_0001.wav
    │   ├── audioRaul_0002.wav
    │   └── ...
    └── metadata.csv
```

El archivo `metadata.csv` sigue el formato:

```
audioRaul_0001.wav|Transcripción del primer segmento.
audioRaul_0002.wav|Transcripción del segundo segmento.
```

> Se recomienda entre 30 y 120 minutos de audio para obtener resultados de calidad.

---

## Uso

### Entrenar una voz

```bash
python3 speaker.py -t audioRaul -n Raul
```

```bash
# Con parámetros opcionales
python3 speaker.py -t audioRaul -n Raul --lang es --epochs 10 --batch 4
```

| Flag | Descripción |
|---|---|
| `-t, --train` | Nombre del dataset en `datasets/` |
| `-n, --name` | Nombre identificador de la voz entrenada |
| `--lang` | Código de idioma (default: `es`) |
| `--epochs` | Número de épocas (default: `10`) |
| `--batch` | Batch size (default: `4`) |

### Generar audio

```bash
python3 speaker.py -g Raul texto.txt -o salida.wav
```

```bash
# Sin especificar nombre de salida (usa nombre_voz + timestamp)
python3 speaker.py -g Raul texto.txt
```

| Flag | Descripción |
|---|---|
| `-g, --generate` | Nombre de la voz fine-tuned a usar |
| `-o, --output` | Nombre del archivo WAV de salida (opcional) |
| `--speed` | Velocidad del habla, 0.5–2.0 (default: `1.0`) |
| `--temperature` | Temperatura del GPT, 0.1–1.0 (default: `0.75`) |

### Gestión de voces

```bash
# Listar todas las voces entrenadas disponibles
python3 speaker.py --list

# Eliminar una voz (pide confirmación)
python3 speaker.py --delete Raul

# Eliminar sin confirmación
python3 speaker.py --delete-force Raul
```

---

## Estructura del proyecto

```
SpeakerAI/
│
├── datasets/                     # Datos de entrenamiento
│
├── configs/
│   └── xtts_train_template.json  # Plantilla de configuración del entrenamiento
│
├── models/
│   ├── base/                     # Modelo preentrenado XTTS v2
│   └── trained/                  # Voces fine-tuned
│       └── Raul/
│           ├── model.pth
│           ├── config.json
│           ├── vocab.json
│           └── speaker_embedding.pth
│
├── training/                     # Artefactos temporales de entrenamiento
│   └── Raul/
│       ├── logs/                 # TensorBoard logs
│       ├── checkpoints/          # Checkpoints intermedios
│       └── config_train.json     # Config usada en este run
│
├── core/
│   ├── dataset_loader.py         # Validación del dataset
│   ├── trainer.py                # Orquestación del fine-tuning
│   ├── model_manager.py          # Gestión de modelos y embeddings
│   └── synthesizer.py            # Generación de audio
│
├── tools/
│   └── splitter.py               # Segmentación y transcripción de audio
│
└── speaker.py                    # CLI principal
```

---

## Monitorización del entrenamiento

Durante el entrenamiento puedes monitorizar la curva de loss en tiempo real con TensorBoard:

```bash
tensorboard --logdir training/Raul/logs
```

Abre `http://localhost:6006` en tu navegador.

---

## Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| `coqui-tts` | ≥ 0.27.0 | Motor TTS y GPTTrainer |
| `torch` | CUDA 12.x | Deep learning |
| `transformers` | ≥ 4.47.0, < 5.0 | Componentes internos de XTTS |
| `librosa` | ≥ 0.11.0 | Análisis y validación de audio |
| `soundfile` | ≥ 0.12.0 | Escritura de WAV de salida |
| `pysbd` | ≥ 0.3.4 | Segmentación de texto en frases |

> **Nota:** `transformers >= 5.0` elimina `isin_mps_friendly`, usado internamente por `coqui-tts`. Mantener `< 5.0` es obligatorio.

---

## Licencia

El modelo base XTTS v2 está bajo la [Coqui Public Model License (CPML)](https://coqui.ai/cpml), que permite uso personal y de investigación. El código de SpeakerAI es de uso libre para fines no comerciales.

---

<div align="center">

**Created by AdZiBo**

[![GitHub](https://img.shields.io/badge/GitHub-adzibo-181717?style=flat&logo=github)](https://github.com/adzibo)

</div>
