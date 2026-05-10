#!/usr/bin/env python3
"""
splitter.py
===========
Herramienta de preparación de datasets para fine-tuning de XTTS v2 (Coqui TTS).

Recibe un archivo de audio tratado (WAV), lo segmenta en fragmentos óptimos
y genera transcripciones automáticas usando Whisper. El resultado es la
estructura de carpetas y el archivo metadata.csv necesarios para entrenar
un modelo de voz con XTTS v2.

Uso básico:
    python3 splitter.py audioTratado.wav -o /ruta/al/dataset/MiVoz

Uso avanzado:
    python3 splitter.py voz.wav -o datasets/MiVoz --model large-v3 --lang es --silence-thresh -35

Creado por: AdZiBo
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS ESTÁNDAR
#  Módulos que vienen incluidos con Python, no requieren instalación adicional.
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import csv
import re
import json
import argparse
import logging
import warnings
from pathlib import Path

# Suprimimos advertencias que no aportan información útil al administrador
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
#  COMPROBACIÓN DE DEPENDENCIAS EXTERNAS
#
#  Intentamos importar cada librería necesaria. Si alguna falta, la añadimos
#  a la lista MISSING_DEPS y al final mostramos un mensaje de error claro
#  indicando cómo instalarlas. Así evitamos errores crípticos en tiempo de
#  ejecución y facilitamos la configuración del entorno.
# ──────────────────────────────────────────────────────────────────────────────

MISSING_DEPS = []  # Lista que acumula los nombres de dependencias no encontradas

try:
    import numpy as np                          # Operaciones numéricas sobre arrays de audio
except ImportError:
    MISSING_DEPS.append("numpy")

try:
    import soundfile as sf                      # Lectura de metadatos de archivos WAV (ej: duración)
except ImportError:
    MISSING_DEPS.append("soundfile")

try:
    from pydub import AudioSegment              # Carga, manipulación y exportación de audio
    from pydub.silence import detect_nonsilent  # Detección de zonas con voz (no silencio)
except ImportError:
    MISSING_DEPS.append("pydub")

try:
    import whisper                              # Modelo de transcripción automática de OpenAI
except ImportError:
    MISSING_DEPS.append("openai-whisper")

try:
    from tqdm import tqdm                       # Barras de progreso en consola
except ImportError:
    # Si tqdm no está instalado, definimos una clase mínima que no hace nada,
    # para que el resto del código funcione igualmente sin barras de progreso.
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable or []

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self): return self

        def __exit__(self, *a): pass

        def set_description(self, *a): pass

        def update(self, *a): pass

# Si hay dependencias que faltan, mostramos el comando de instalación y salimos
if MISSING_DEPS:
    print("\n❌  Faltan dependencias. Instálalas con:\n")
    pip_names = {
        "numpy":          "numpy",
        "soundfile":      "soundfile",
        "pydub":          "pydub",
        "openai-whisper": "openai-whisper",
    }
    cmds = " ".join(pip_names[d] for d in MISSING_DEPS if d in pip_names)
    print(f"    pip install {cmds}\n")
    print("También necesitas ffmpeg instalado en el sistema:")
    print("    sudo apt install ffmpeg\n")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTES DE CONFIGURACIÓN
#
#  Estos valores definen el comportamiento por defecto de la herramienta.
#  Están centralizados aquí para facilitar su ajuste sin tener que buscarlos
#  dispersos por el código.
#
#  Parámetros de XTTS v2:
#    - MIN/MAX DURATION: rango válido de duración de cada segmento de audio
#    - TARGET_SR: frecuencia de muestreo requerida por XTTS v2 (22.050 Hz)
#
#  Parámetros de segmentación por silencio:
#    - SILENCE_THRESH_DB: umbral relativo (en dB) para considerar silencio
#    - MIN_SILENCE_MS: duración mínima (ms) para que una pausa cuente como silencio
#    - KEEP_SILENCE_MS: margen de silencio que se mantiene a cada lado del segmento
#
#  Parámetros de Whisper:
#    - WHISPER_MODEL: tamaño del modelo (cuanto más grande, más preciso pero más lento)
#    - LANGUAGE: código ISO 639-1 del idioma del audio
# ──────────────────────────────────────────────────────────────────────────────

XTTS_MIN_DURATION_S = 1.5           # Duración mínima de un segmento válido (segundos)
XTTS_MAX_DURATION_S = 11.0          # Duración máxima de un segmento válido (segundos)
XTTS_TARGET_SR = 22050              # Sample rate objetivo para XTTS v2 (Hz)

DEFAULT_SILENCE_THRESH_DB = -40     # Umbral de silencio relativo al volumen medio del audio (dB)
DEFAULT_MIN_SILENCE_MS = 400        # Pausa mínima para considerar separación entre frases (ms)
DEFAULT_KEEP_SILENCE_MS = 80        # Margen de silencio que se conserva en los extremos (ms)

DEFAULT_WHISPER_MODEL = "large-v3"  # Modelo Whisper por defecto (mejor calidad disponible)
DEFAULT_LANGUAGE = "es"             # Idioma por defecto: español


# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE LOGGING
#
#  Usamos el módulo estándar `logging` para registrar mensajes informativos,
#  advertencias y errores durante la ejecución. El formato incluye la hora,
#  el nivel del mensaje y el texto, lo que facilita el seguimiento del proceso.
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("splitter")  # Logger con nombre propio para identificar la fuente


# ─── BANNER ───────────────────────────────────────────────────────────────────
BANNER = r"""
  ███████╗██████╗ ██╗     ██╗████████╗████████╗███████╗██████╗
  ██╔════╝██╔══██╗██║     ██║╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗
  ███████╗██████╔╝██║     ██║   ██║      ██║   █████╗  ██████╔╝
  ╚════██║██╔═══╝ ██║     ██║   ██║      ██║   ██╔══╝  ██╔══██╗
  ███████║██║     ███████╗██║   ██║      ██║   ███████╗██║  ██║
  ╚══════╝╚═╝     ╚══════╝╚═╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝
                                               Created by AdZiBo
"""


def print_banner():
    """Imprime el banner de presentación de la herramienta."""
    print(BANNER)


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: load_audio
#
#  Carga el archivo WAV de entrada y muestra información básica sobre él.
#  Si el archivo no existe o no es WAV, el programa termina con un mensaje
#  de error claro.
# ──────────────────────────────────────────────────────────────────────────────
def load_audio(wav_path: Path) -> AudioSegment:
    """
    Carga el archivo WAV desde disco y devuelve un objeto AudioSegment.

    Args:
        wav_path (Path): Ruta al archivo WAV de entrada.

    Returns:
        AudioSegment: El audio cargado en memoria.
    """
    log.info(f"Cargando audio: {wav_path}")

    # Verificamos que el archivo existe antes de intentar abrirlo
    if not wav_path.exists():
        log.error(f"No se encontró el archivo: {wav_path}")
        sys.exit(1)

    # Solo aceptamos WAV; otros formatos requieren conversión previa
    if wav_path.suffix.lower() != ".wav":
        log.error("El archivo debe estar en formato WAV.")
        sys.exit(1)

    audio = AudioSegment.from_wav(str(wav_path))

    # Mostramos un resumen técnico del audio para que el administrador
    # pueda verificar que se ha cargado correctamente
    duration_min = len(audio) / 60_000  # len() devuelve milisegundos
    log.info(
        f"Audio cargado: {duration_min:.2f} min | "
        f"{audio.frame_rate} Hz | "
        f"{audio.channels} canal(es) | "
        f"{audio.sample_width * 8}-bit"
    )

    return audio


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: normalize_audio
#
#  Asegura que el audio cumple los requisitos técnicos de XTTS v2:
#    - Mono (1 canal): el modelo no acepta audio estéreo
#    - 22.050 Hz: la frecuencia de muestreo exacta que espera XTTS v2
#
#  Si el audio ya cumple estos requisitos, no se realiza ninguna conversión.
# ──────────────────────────────────────────────────────────────────────────────
def normalize_audio(audio: AudioSegment) -> AudioSegment:
    """
    Convierte el audio a mono y lo remuestrea a la frecuencia requerida por XTTS v2.

    Args:
        audio (AudioSegment): Audio original cargado.

    Returns:
        AudioSegment: Audio normalizado listo para segmentar.
    """
    # Convertir a mono si es estéreo (XTTS v2 solo acepta 1 canal)
    if audio.channels > 1:
        log.info("Convirtiendo a mono...")
        audio = audio.set_channels(1)

    # Ajustar la frecuencia de muestreo si no coincide con la requerida
    if audio.frame_rate != XTTS_TARGET_SR:
        log.info(f"Remuestreando de {audio.frame_rate} Hz a {XTTS_TARGET_SR} Hz...")
        audio = audio.set_frame_rate(XTTS_TARGET_SR)

    return audio


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: segment_audio
#
#  Divide el audio en fragmentos individuales siguiendo estas reglas:
#
#  1. Detección de voz: se identifican las regiones con actividad vocal
#     descartando los silencios largos.
#
#  2. Fusión inteligente: si un fragmento es demasiado corto, se fusiona
#     con el siguiente. Si dos fragmentos cercanos suman menos del máximo
#     permitido, también se fusionan para aprovechar mejor los datos.
#
#  3. División por tamaño: si un segmento fusionado supera el máximo,
#     se divide en partes iguales respetando el límite de duración.
#
#  4. Filtrado final: se descartan los fragmentos que no alcanzan la
#     duración mínima requerida por XTTS v2.
# ──────────────────────────────────────────────────────────────────────────────
def segment_audio(
    audio: AudioSegment,
    silence_thresh_db: int,
    min_silence_ms: int,
    keep_silence_ms: int,
    min_dur_s: float,
    max_dur_s: float,
) -> list:
    """
    Segmenta el audio en fragmentos de duración óptima para XTTS v2.

    Args:
        audio (AudioSegment):    Audio normalizado.
        silence_thresh_db (int): Umbral de silencio relativo en dB (negativo).
        min_silence_ms (int):    Pausa mínima para separar frases (ms).
        keep_silence_ms (int):   Margen de silencio a conservar en los bordes (ms).
        min_dur_s (float):       Duración mínima de un segmento válido (segundos).
        max_dur_s (float):       Duración máxima de un segmento (segundos).

    Returns:
        list[AudioSegment]: Lista de segmentos de audio válidos.
    """
    log.info("Detectando silencios y segmentando...")

    # Convertimos duraciones de segundos a milisegundos para trabajar con pydub
    min_dur_ms = int(min_dur_s * 1000)
    max_dur_ms = int(max_dur_s * 1000)

    # Detectamos las regiones donde hay voz (zonas no silenciosas).
    # El umbral se calcula de forma relativa al volumen medio (dBFS) del audio,
    # lo que lo hace robusto ante audios con diferentes niveles de grabación.
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=audio.dBFS + silence_thresh_db,
        seek_step=10,  # Resolución de análisis en ms (10ms es un buen equilibrio)
    )

    # Si no se detecta ninguna zona de voz, algo va mal (archivo silencioso, umbral incorrecto...)
    if not nonsilent_ranges:
        log.error("No se detectó voz en el audio. Revisa los parámetros de silencio.")
        sys.exit(1)

    log.info(f"Detectadas {len(nonsilent_ranges)} regiones de voz.")

    # ── Paso 1: Añadir margen de silencio a cada región detectada ──────────
    # Expandimos ligeramente cada región para no cortar la voz en seco
    audio_len = len(audio)
    segments_ms = []
    for start, end in nonsilent_ranges:
        s = max(0, start - keep_silence_ms)          # No salir del inicio del audio
        e = min(audio_len, end + keep_silence_ms)     # No salir del final del audio
        segments_ms.append((s, e))

    # ── Paso 2: Fusionar fragmentos cortos o cercanos ──────────────────────
    # Recorremos los segmentos y decidimos si fusionarlos con el buffer acumulado
    merged = []
    buf_start, buf_end = segments_ms[0]  # Iniciamos el buffer con el primer segmento

    for start, end in segments_ms[1:]:
        buf_dur = buf_end - buf_start    # Duración actual del buffer
        seg_dur = end - start            # Duración del segmento candidato

        if buf_dur < min_dur_ms:
            # El buffer es demasiado corto → fusionamos para ganar duración
            buf_end = end

        elif buf_dur + seg_dur <= max_dur_ms:
            # La fusión no superaría el máximo → valoramos si la pausa entre ellos es pequeña
            gap = start - buf_end
            if gap < min_silence_ms * 1.5:
                # La pausa es corta → fusionamos (misma frase o frase relacionada)
                buf_end = end
            else:
                # La pausa es larga → los guardamos por separado
                merged.append((buf_start, buf_end))
                buf_start, buf_end = start, end
        else:
            # Fusionar superaría el máximo → guardamos el buffer y empezamos uno nuevo
            merged.append((buf_start, buf_end))
            buf_start, buf_end = start, end

    merged.append((buf_start, buf_end))  # Guardamos el último buffer pendiente

    # ── Paso 3: Dividir segmentos que superan la duración máxima ──────────
    # Si tras la fusión algún segmento es demasiado largo, lo partimos en trozos iguales
    final_segments = []
    for start, end in merged:
        dur = end - start
        if dur > max_dur_ms:
            # Dividimos en chunks consecutivos del tamaño máximo
            pos = start
            while pos < end:
                chunk_end = min(pos + max_dur_ms, end)
                final_segments.append(audio[pos:chunk_end])
                pos = chunk_end
        else:
            final_segments.append(audio[start:end])

    # ── Paso 4: Filtrar segmentos demasiado cortos ─────────────────────────
    valid = [s for s in final_segments if len(s) >= min_dur_ms]
    skipped = len(final_segments) - len(valid)
    if skipped:
        log.warning(f"Se descartaron {skipped} segmentos por ser demasiado cortos.")

    log.info(f"Segmentos válidos generados: {len(valid)}")
    return valid


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: export_wavs
#
#  Guarda cada segmento de audio como un archivo WAV individual en la carpeta
#  de salida. Los nombres siguen el patrón: {base_name}_{número:4 dígitos}.wav
#  (ej: miVoz_0001.wav, miVoz_0002.wav, ...)
# ──────────────────────────────────────────────────────────────────────────────
def export_wavs(segments: list, wavs_dir: Path, base_name: str) -> list:
    """
    Exporta la lista de segmentos de audio como archivos WAV individuales.

    Args:
        segments (list[AudioSegment]): Segmentos de audio a exportar.
        wavs_dir (Path):               Directorio donde se guardarán los WAVs.
        base_name (str):               Prefijo para los nombres de archivo.

    Returns:
        list[str]: Lista de nombres de archivo exportados (solo el nombre, sin ruta).
    """
    # Creamos el directorio de salida si no existe (incluyendo directorios intermedios)
    wavs_dir.mkdir(parents=True, exist_ok=True)
    filenames = []

    log.info(f"Exportando {len(segments)} segmentos WAV a: {wavs_dir}")

    for i, seg in enumerate(tqdm(segments, desc="Exportando WAVs"), start=1):
        # Nombre con índice de 4 dígitos para mantener orden alfabético correcto
        fname = f"{base_name}_{i:04d}.wav"
        fpath = wavs_dir / fname
        seg.export(str(fpath), format="wav")
        filenames.append(fname)

    return filenames


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: load_whisper_model
#
#  Carga el modelo de transcripción Whisper en memoria. La primera vez que
#  se usa un modelo, Whisper lo descarga automáticamente (~1-3 GB según el
#  tamaño). Las ejecuciones posteriores lo cargan desde caché local.
# ──────────────────────────────────────────────────────────────────────────────
def load_whisper_model(model_name: str):
    """
    Carga el modelo Whisper especificado en memoria.

    Args:
        model_name (str): Nombre del modelo Whisper (ej: "large-v3", "medium").

    Returns:
        El modelo Whisper listo para transcribir.
    """
    log.info(f"Cargando modelo Whisper '{model_name}' (puede tardar la primera vez)...")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        log.error(f"Error al cargar el modelo Whisper: {e}")
        sys.exit(1)

    log.info("Modelo Whisper cargado correctamente.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: transcribe_segments
#
#  Transcribe cada segmento WAV usando Whisper y devuelve un diccionario
#  {nombre_archivo: texto_transcrito}. Los segmentos con transcripción vacía
#  o que producen un error se omiten y se eliminan del dataset.
#
#  Parámetros clave de Whisper:
#    - beam_size / best_of: controlan la calidad de decodificación (más alto = más preciso)
#    - temperature: 0.0 = resultado determinista y más exacto
#    - condition_on_previous_text: False evita que errores anteriores contagien al siguiente
# ──────────────────────────────────────────────────────────────────────────────
def transcribe_segments(
    model,
    wavs_dir: Path,
    filenames: list,
    language: str,
    beam_size: int,
    best_of: int,
    temperature: float,
) -> dict:
    """
    Transcribe todos los segmentos WAV con el modelo Whisper cargado.

    Args:
        model:           Modelo Whisper ya cargado en memoria.
        wavs_dir (Path): Directorio donde están los WAVs exportados.
        filenames (list[str]): Nombres de los archivos WAV a transcribir.
        language (str):  Código de idioma ISO 639-1 (ej: "es", "en").
        beam_size (int): Tamaño del haz de búsqueda para decodificación.
        best_of (int):   Número de candidatos a evaluar por segmento.
        temperature (float): Temperatura de muestreo (0.0 = greedy/determinista).

    Returns:
        dict: Diccionario {nombre_archivo: transcripción_limpia}.
    """
    transcriptions = {}
    log.info(f"Transcribiendo {len(filenames)} segmentos con Whisper...")
    log.info(f"  Idioma: {language} | beam_size: {beam_size} | best_of: {best_of}")

    skipped = 0  # Contador de segmentos omitidos por error o transcripción vacía

    for fname in tqdm(filenames, desc="Transcribiendo"):
        fpath = str(wavs_dir / fname)

        try:
            result = model.transcribe(
                fpath,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                condition_on_previous_text=False,  # Cada segmento es independiente
                word_timestamps=False,             # No necesitamos timestamps por palabra
                fp16=False,                        # fp16 puede fallar en CPUs; fp32 es más seguro
                verbose=False,                     # Evitamos logs internos de Whisper
            )

            # Extraemos y limpiamos el texto resultante
            text = result["text"].strip()
            text = clean_transcription(text)

            # Si tras la limpieza el texto queda vacío, descartamos el segmento
            if not text:
                log.warning(f"Transcripción vacía para {fname}, se omitirá.")
                skipped += 1
                continue

            transcriptions[fname] = text

        except Exception as e:
            log.warning(f"Error transcribiendo {fname}: {e}")
            skipped += 1

    if skipped:
        log.warning(f"Se omitieron {skipped} segmentos (error o transcripción vacía).")

    log.info(f"Transcripciones completadas: {len(transcriptions)}/{len(filenames)}")
    return transcriptions


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: clean_transcription
#
#  Limpia el texto devuelto por Whisper eliminando:
#    - Anotaciones entre corchetes: [música], [ruido], [aplausos]...
#    - Anotaciones entre paréntesis: (risas), (inaudible)...
#    - Texto en negrita de Markdown: *texto*
#    - Caracteres especiales no esperados en texto hablado
#    - Espacios duplicados
#
#  Se conservan letras, números, puntuación básica y caracteres del español.
# ──────────────────────────────────────────────────────────────────────────────
def clean_transcription(text: str) -> str:
    """
    Elimina artefactos y caracteres no deseados de una transcripción de Whisper.

    Args:
        text (str): Texto bruto devuelto por Whisper.

    Returns:
        str: Texto limpio listo para incluir en metadata.csv.
    """
    # Eliminar anotaciones de eventos no verbales: [música], [ruido de fondo], etc.
    text = re.sub(r"\[.*?\]", "", text)

    # Eliminar anotaciones entre paréntesis: (risas), (inaudible), etc.
    text = re.sub(r"\(.*?\)", "", text)

    # Eliminar énfasis de Markdown: *palabra*
    text = re.sub(r"\*.*?\*", "", text)

    # Conservar solo caracteres válidos en texto hablado en español:
    # letras, dígitos, espacios, puntuación básica y caracteres con tilde
    text = re.sub(
        r"[^\w\s.,;:!?¿¡\-'\"áéíóúüñÁÉÍÓÚÜÑ]",
        "",
        text,
        flags=re.UNICODE
    )

    # Normalizar espacios múltiples a uno solo
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: write_metadata
#
#  Genera el archivo metadata.csv en el directorio de salida del dataset.
#  El formato es el estándar que espera XTTS v2:
#
#      nombre_archivo.wav|Texto de la transcripción
#
#  Una línea por segmento, sin cabecera, separado por "|".
# ──────────────────────────────────────────────────────────────────────────────
def write_metadata(output_dir: Path, transcriptions: dict) -> Path:
    """
    Escribe el archivo metadata.csv con el formato requerido por XTTS v2.

    Args:
        output_dir (Path):      Directorio raíz del dataset.
        transcriptions (dict):  Diccionario {nombre_wav: texto_transcrito}.

    Returns:
        Path: Ruta al archivo metadata.csv generado.
    """
    metadata_path = output_dir / "metadata.csv"
    log.info(f"Generando metadata.csv: {metadata_path}")

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        for fname, text in transcriptions.items():
            f.write(f"{fname}|{text}\n")

    log.info(f"metadata.csv generado con {len(transcriptions)} entradas.")
    return metadata_path


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: print_summary
#
#  Muestra un resumen final en consola con las estadísticas del dataset
#  generado: número de segmentos, duración total, longitud media de las
#  transcripciones y rutas de los archivos de salida.
#
#  También evalúa si la cantidad de datos es suficiente para obtener un
#  buen resultado en el fine-tuning de XTTS v2:
#    - < 5 min  → advertencia, dataset muy pequeño
#    - 5-30 min → aceptable pero mejorable
#    - > 30 min → óptimo
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(output_dir: Path, wavs_dir: Path, transcriptions: dict, total_segments: int):
    """
    Muestra un resumen completo del dataset generado al finalizar el proceso.

    Args:
        output_dir (Path):      Directorio raíz del dataset.
        wavs_dir (Path):        Directorio con los archivos WAV exportados.
        transcriptions (dict):  Transcripciones válidas generadas.
        total_segments (int):   Total de segmentos antes del filtrado.
    """
    # Calculamos la duración total sumando la duración de cada WAV válido
    total_duration_s = 0.0
    for fname in transcriptions:
        fpath = wavs_dir / fname
        try:
            info = sf.info(str(fpath))
            total_duration_s += info.duration
        except Exception:
            pass  # Si un archivo falla, lo ignoramos en el cálculo

    # Longitud media de las transcripciones (en caracteres)
    avg_len = (
        sum(len(t) for t in transcriptions.values()) / len(transcriptions)
        if transcriptions else 0
    )

    # ── Resumen de resultados ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅  DATASET GENERADO CORRECTAMENTE")
    print("═" * 60)
    print(f"  📁  Directorio      : {output_dir}")
    print(f"  🎙  Segmentos        : {len(transcriptions)} / {total_segments} válidos")
    print(f"  ⏱  Duración total   : {total_duration_s / 60:.2f} min ({total_duration_s:.1f}s)")
    print(f"  📝  Chars/seg (avg) : {avg_len:.1f}")
    print(f"  📄  metadata.csv    : {output_dir / 'metadata.csv'}")
    print(f"  📂  wavs/           : {wavs_dir}")
    print("═" * 60)

    # ── Evaluación de la cantidad de datos ───────────────────────────────
    if total_duration_s < 300:
        print("  ⚠️  Advertencia: menos de 5 min de audio.")
        print("     XTTS v2 mejora notablemente con >30 min de datos.")
    elif total_duration_s < 1800:
        print("  ℹ️  Dataset pequeño. Recomendado: >30 min para mejor calidad.")
    else:
        print("  🎉  ¡Cantidad de datos excelente para fine-tuning!")

    print()


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN: build_parser
#
#  Define los argumentos que acepta el script por línea de comandos usando
#  argparse. Los argumentos están agrupados por categoría para mayor claridad:
#
#    Posicional:
#      audio         → Ruta al archivo WAV de entrada (obligatorio)
#
#    Opciones principales:
#      -o / --output → Directorio de salida del dataset (obligatorio)
#      --base-name   → Prefijo para los nombres de los WAVs exportados
#
#    Grupo "Transcripción":
#      --model       → Tamaño del modelo Whisper
#      --lang        → Idioma del audio
#      --beam-size   → Precisión de decodificación de Whisper
#      --best-of     → Candidatos evaluados por segmento
#      --temperature → Creatividad de la transcripción (0 = exacto)
#
#    Grupo "Segmentación":
#      --silence-thresh   → Umbral de detección de silencio (dB)
#      --min-silence-ms   → Duración mínima de silencio entre frases
#      --keep-silence-ms  → Margen de silencio a conservar en los bordes
#      --min-dur          → Duración mínima de segmento (segundos)
#      --max-dur          → Duración máxima de segmento (segundos)
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """
    Construye y devuelve el parser de argumentos de línea de comandos.

    Returns:
        argparse.ArgumentParser: Parser configurado con todos los argumentos.
    """
    p = argparse.ArgumentParser(
        prog="splitter.py",
        description=(
            "Prepara un audio tratado (WAV) para fine-tuning de XTTS v2.\n"
            "Segmenta el audio y genera metadata.csv con transcripciones de alta precisión."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Ejemplos de uso:
          python3 splitter.py voz.wav -o datasets/MiVoz
          python3 splitter.py voz.wav -o datasets/MiVoz --model large-v3 --lang es
          python3 splitter.py voz.wav -o datasets/MiVoz --silence-thresh -35 --max-dur 9.0
                """,
    )

    # ── Argumento posicional obligatorio ─────────────────────────────────
    p.add_argument(
        "audio",
        type=Path,
        help="Ruta al archivo de audio tratado en formato WAV."
    )

    # ── Directorio de salida (obligatorio) ────────────────────────────────
    p.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Directorio donde se guardará el dataset generado."
    )

    # ── Grupo: opciones de transcripción con Whisper ──────────────────────
    g = p.add_argument_group("Transcripción (Whisper)")
    g.add_argument(
        "--model",
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Modelo Whisper a usar. Más grande = más preciso pero más lento. (default: {DEFAULT_WHISPER_MODEL})"
    )
    g.add_argument(
        "--lang",
        default=DEFAULT_LANGUAGE,
        help=f"Código ISO 639-1 del idioma del audio. (default: '{DEFAULT_LANGUAGE}')"
    )
    g.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Tamaño del haz de búsqueda de Whisper. Mayor = más preciso. (default: 5)"
    )
    g.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Candidatos evaluados por segmento en Whisper. (default: 5)"
    )
    g.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de muestreo de Whisper. 0.0 = determinista. (default: 0.0)"
    )

    # ── Grupo: opciones de segmentación por silencio ─────────────────────
    g2 = p.add_argument_group("Segmentación")
    g2.add_argument(
        "--silence-thresh",
        type=int,
        default=DEFAULT_SILENCE_THRESH_DB,
        help=f"Umbral de silencio relativo en dB (valor negativo). (default: {DEFAULT_SILENCE_THRESH_DB})"
    )
    g2.add_argument(
        "--min-silence-ms",
        type=int,
        default=DEFAULT_MIN_SILENCE_MS,
        help=f"Duración mínima de silencio para separar frases (ms). (default: {DEFAULT_MIN_SILENCE_MS})"
    )
    g2.add_argument(
        "--keep-silence-ms",
        type=int,
        default=DEFAULT_KEEP_SILENCE_MS,
        help=f"Margen de silencio a conservar en los bordes del segmento (ms). (default: {DEFAULT_KEEP_SILENCE_MS})"
    )
    g2.add_argument(
        "--min-dur",
        type=float,
        default=XTTS_MIN_DURATION_S,
        help=f"Duración mínima de un segmento válido en segundos. (default: {XTTS_MIN_DURATION_S})"
    )
    g2.add_argument(
        "--max-dur",
        type=float,
        default=XTTS_MAX_DURATION_S,
        help=f"Duración máxima de un segmento en segundos. (default: {XTTS_MAX_DURATION_S})"
    )

    # ── Nombre base para los archivos WAV exportados ──────────────────────
    p.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Prefijo para nombrar los archivos WAV. Por defecto usa el nombre del audio de entrada."
    )

    return p


# ──────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL: main
#
#  Orquesta todo el flujo de trabajo:
#
#  1. Parsear argumentos de línea de comandos
#  2. Mostrar banner y configuración activa
#  3. Cargar y normalizar el audio de entrada
#  4. Segmentar el audio en fragmentos óptimos
#  5. Exportar los segmentos como archivos WAV
#  6. Cargar el modelo Whisper
#  7. Transcribir todos los segmentos
#  8. Eliminar WAVs sin transcripción válida
#  9. Escribir metadata.csv
#  10. Mostrar resumen final
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Punto de entrada principal. Ejecuta el pipeline completo de preparación del dataset."""

    # ── 1. Parsear argumentos ─────────────────────────────────────────────
    parser = build_parser()
    args = parser.parse_args()

    # Resolvemos las rutas a absolutas para evitar ambigüedades
    audio_path: Path = args.audio.resolve()
    output_dir: Path = args.output.resolve()
    wavs_dir:   Path = output_dir / "wavs"  # Subcarpeta estándar para los WAVs del dataset

    # El prefijo de nombre es el nombre del archivo de audio si no se especifica otro
    base_name = args.base_name or audio_path.stem

    # ── 2. Mostrar banner y configuración activa ──────────────────────────
    print_banner()
    print(f"  Audio entrada  : {audio_path}")
    print(f"  Directorio     : {output_dir}")
    print(f"  Modelo Whisper : {args.model}")
    print(f"  Idioma         : {args.lang}")
    print()

    # ── 3. Cargar y normalizar el audio ───────────────────────────────────
    audio = load_audio(audio_path)
    audio = normalize_audio(audio)

    # ── 4. Segmentar el audio ─────────────────────────────────────────────
    segments = segment_audio(
        audio,
        silence_thresh_db=args.silence_thresh,
        min_silence_ms=args.min_silence_ms,
        keep_silence_ms=args.keep_silence_ms,
        min_dur_s=args.min_dur,
        max_dur_s=args.max_dur,
    )

    total_segments = len(segments)
    if total_segments == 0:
        log.error("No se generaron segmentos. Ajusta los parámetros de segmentación.")
        sys.exit(1)

    # ── 5. Exportar segmentos como archivos WAV ───────────────────────────
    filenames = export_wavs(segments, wavs_dir, base_name)

    # ── 6. Cargar modelo Whisper ──────────────────────────────────────────
    model = load_whisper_model(args.model)

    # ── 7. Transcribir todos los segmentos ────────────────────────────────
    transcriptions = transcribe_segments(
        model,
        wavs_dir,
        filenames,
        language=args.lang,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
    )

    # ── 8. Eliminar WAVs sin transcripción válida ─────────────────────────
    # Si un segmento no pudo transcribirse, lo borramos del disco para
    # mantener la coherencia entre los WAVs y el metadata.csv
    valid_filenames = set(transcriptions.keys())
    for fname in filenames:
        if fname not in valid_filenames:
            fpath = wavs_dir / fname
            if fpath.exists():
                fpath.unlink()
                log.debug(f"Eliminado WAV sin transcripción: {fname}")

    # ── 9. Escribir metadata.csv ──────────────────────────────────────────
    write_metadata(output_dir, transcriptions)

    # ── 10. Mostrar resumen final ─────────────────────────────────────────
    print_summary(output_dir, wavs_dir, transcriptions, total_segments)


# ──────────────────────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
#
#  Esta comprobación asegura que main() solo se ejecuta cuando el script
#  se llama directamente (python3 splitter.py ...) y no cuando se importa
#  como módulo desde otro script Python.
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
