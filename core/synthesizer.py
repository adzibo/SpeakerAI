"""
core/synthesizer.py
-------------------
Generación de audio TTS usando una voz fine-tuned de SpeakerTTS.

Pipeline:
  1. Carga el modelo fine-tuned (GPT encoder) desde models/trained/<name>/.
  2. El decoder y el DVAE se cargan desde models/base/ (no se fine-tunean).
  3. Recupera el speaker_embedding cacheado. Si no existe, lo calcula al vuelo
     usando los WAVs del dataset de referencia.
  4. Lee el texto de entrada (archivo .txt o string directo).
  5. Segmenta el texto en frases con pysbd para evitar artefactos en textos largos.
  6. Genera el audio frase a frase y concatena.
  7. Guarda el WAV resultante en la ruta de salida especificada.

Uso (desde speaker.py):
    from core.synthesizer import Synthesizer
    synth = Synthesizer(voice_name="Adil", project_root=".")
    synth.generate(text_source="descripcion.txt", output_path="salida.wav")
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from core.model_manager import ModelManager


# Parámetros de inferencia por defecto.
# Estos valores equilibran calidad y velocidad para uso en producción.
DEFAULT_LANGUAGE = "es"
DEFAULT_TEMPERATURE = 0.75          # 0.1 = más determinista, 1.0 = más expresivo
DEFAULT_REPETITION_PENALTY = 5.0    # penaliza repeticiones de tokens de audio
DEFAULT_TOP_K = 50                  # muestrea entre los 50 tokens más probables
DEFAULT_TOP_P = 0.85                # nucleus sampling: acumula hasta 85% de probabilidad
DEFAULT_SPEED = 1.0                 # velocidad del habla (rango recomendado: 0.5–2.0)
DEFAULT_NUM_GPT_OUTPUTS = 1         # 1 = rápido; aumentar para reranking y más calidad
OUTPUT_SAMPLE_RATE = 24000          # Hz — frecuencia de salida de XTTS v2 (fija)


class Synthesizer:
    """Genera audio TTS con una voz fine-tuned de SpeakerTTS."""

    def __init__(
        self,
        voice_name: str,
        project_root: str = ".",
        language: str = DEFAULT_LANGUAGE,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        speed: float = DEFAULT_SPEED,
    ):
        self.voice_name = voice_name
        self.language = language
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.speed = speed

        self.root = Path(project_root).resolve()
        self.mm = ModelManager(project_root)

        # El modelo se carga de forma lazy en la primera llamada a generate(),
        # evitando importar Coqui TTS hasta que sea estrictamente necesario.
        self._model = None
        self._device = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(
        self,
        text_source: str,
        output_path: Optional[str] = None,
    ) -> Path:
        """
        Genera un WAV a partir de text_source (ruta a .txt o string directo).

        Parámetros:
            text_source  -- ruta a un archivo .txt O texto plano directamente.
            output_path  -- ruta del WAV de salida. Si es None, usa el nombre
                            de la voz + timestamp en el directorio actual.

        Devuelve la ruta absoluta al WAV generado.
        """
        print(f"\n{'='*60}")
        print(f"  SpeakerTTS — Generación de audio")
        print(f"  Voz    : {self.voice_name}")
        print(f"  Idioma : {self.language}")
        print(f"{'='*60}\n")

        # 1) Verificar que la voz existe y es válida
        if not self.mm.voice_exists(self.voice_name):
            raise ValueError(
                f"Voz '{self.voice_name}' no encontrada. "
                "Voces disponibles: python3 speaker.py --list"
            )
        if not self.mm.is_voice_valid(self.voice_name):
            raise ValueError(
                f"Voz '{self.voice_name}' incompleta (faltan ficheros). "
                "Re-entrena con: python3 speaker.py -t <dataset> -n {self.voice_name}"
            )

        # 2) Leer texto
        text = self._read_text(text_source)
        if not text.strip():
            raise ValueError("El texto de entrada está vacío.")

        print(f"  Texto  : {len(text)} caracteres")

        # 3) Cargar modelo (lazy)
        self._load_model()

        # 4) Obtener speaker embedding (cacheado o calculado al vuelo)
        gpt_cond_latent, speaker_embedding = self._get_speaker_embedding()

        # 5) Segmentar texto en frases
        sentences = self._split_text(text)
        print(f"  Frases : {len(sentences)} segmentos")

        # 6) Generar audio frase a frase
        audio_chunks = self._synthesize_sentences(
            sentences, gpt_cond_latent, speaker_embedding
        )

        # 7) Concatenar y guardar
        output_path = self._save_audio(audio_chunks, output_path)

        print(f"\n  ✓ Audio generado: {output_path}\n")
        return output_path

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Carga el modelo fine-tuned en memoria (GPU si está disponible, CPU si no).
        Solo se ejecuta la primera vez; las llamadas sucesivas son no-op.
        Los pesos del GPT encoder vienen de models/trained/<voz>/model.pth;
        el decoder y el DVAE se leen de models/base/ (no se fine-tunean).
        """
        if self._model is not None:
            return  # ya cargado

        print("  Cargando modelo...", end=" ", flush=True)

        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as e:
            raise ImportError(
                f"Coqui TTS no está instalado: {e}\n"
                "Ejecuta: pip install coqui-tts"
            ) from e

        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        voice_dir = self.mm.get_voice_path(self.voice_name)
        base_dir = self.mm.get_base_dir()

        config_path = voice_dir / "config.json"
        model_path = voice_dir / "model.pth"
        vocab_path = voice_dir / "vocab.json"

        config = XttsConfig()
        config.load_json(str(config_path))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(model_path),
            vocab_path=str(vocab_path),
            eval=True,
        )

        model.to(self._device)

        self._model = model
        print(f"OK ({self._device.upper()})")

    # ------------------------------------------------------------------
    # Speaker Embedding
    # ------------------------------------------------------------------

    def _get_speaker_embedding(self):
        """
        Devuelve (gpt_cond_latent, speaker_embedding).
        Prioridad: cache en disco → cálculo desde WAVs del dataset.
        """
        import torch

        cached = self.mm.load_speaker_embedding(self.voice_name)
        if cached is not None:
            print("  Speaker embedding: cargado desde caché ✓")
            gpt_cond_latent = cached["gpt_cond_latent"].to(self._device)
            speaker_embedding = cached["speaker_embedding"].to(self._device)
            return gpt_cond_latent, speaker_embedding

        # Sin cache → calcular el embedding desde los WAVs del dataset original.
        print("  Speaker embedding: calculando desde dataset...", end=" ", flush=True)

        # El dataset puede llamarse igual que la voz o llevar el prefijo "audio".
        # Ejemplo: voz "Adil" → se busca en datasets/Adil/ o datasets/audioAdil/.
        dataset_candidates = [
            self.root / "datasets" / self.voice_name,
            self.root / "datasets" / f"audio{self.voice_name}",
        ]
        wavs_dir = None
        for candidate in dataset_candidates:
            if (candidate / "wavs").exists():
                wavs_dir = candidate / "wavs"
                break

        if wavs_dir is None:
            raise FileNotFoundError(
                f"No se encontró el dataset para la voz '{self.voice_name}' "
                "y tampoco hay speaker_embedding.pth cacheado.\n"
                f"Buscado en: {[str(c) for c in dataset_candidates]}\n"
                "Solución: re-entrena la voz para regenerar el embedding."
            )

        # Se usan hasta 5 WAVs de referencia: suficientes para un buen embedding
        # sin que el cálculo sea demasiado lento.
        ref_wavs = sorted(wavs_dir.glob("*.wav"))[:5]
        if not ref_wavs:
            raise FileNotFoundError(
                f"No hay WAVs en: {wavs_dir}"
            )

        gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
            audio_path=[str(w) for w in ref_wavs]
        )
        print(f"OK ({len(ref_wavs)} WAVs de referencia)")

        # Persistir para la próxima vez
        self.mm.save_speaker_embedding(
            self.voice_name, gpt_cond_latent, speaker_embedding
        )

        return (
            gpt_cond_latent.to(self._device),
            speaker_embedding.to(self._device),
        )

    # ------------------------------------------------------------------
    # Segmentación de texto
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list:
        """
        Segmenta el texto en frases con pysbd.
        Agrupa frases cortas para evitar demasiadas llamadas de inferencia.
        """
        try:
            import pysbd
            segmenter = pysbd.Segmenter(language=self._pysbd_lang(), clean=True)
            raw_sentences = segmenter.segment(text)
        except ImportError:
            # Fallback: split simple por punto/salto de línea
            import re
            raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
            raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

        # Agrupar frases muy cortas (< 20 chars) con la siguiente
        sentences = []
        buffer = ""
        for sentence in raw_sentences:
            buffer = (buffer + " " + sentence).strip()
            if len(buffer) >= 20:
                sentences.append(buffer)
                buffer = ""
        if buffer:
            sentences.append(buffer)

        return [s for s in sentences if s.strip()]

    def _pysbd_lang(self) -> str:
        """Mapea el código de idioma XTTS al de pysbd."""
        mapping = {
            "es": "es", "en": "en", "fr": "fr", "de": "de",
            "it": "it", "pt": "pt", "nl": "nl", "ru": "ru",
            "ja": "ja", "zh-cn": "zh", "ko": "ko",
            "pl": "pl", "tr": "tr", "ar": "ar", "cs": "cs",
            "hu": "hu",
        }
        return mapping.get(self.language, "en")

    # ------------------------------------------------------------------
    # Síntesis frase a frase
    # ------------------------------------------------------------------

    def _synthesize_sentences(
        self,
        sentences: list,
        gpt_cond_latent,
        speaker_embedding,
    ) -> list:
        """
        Genera audio para cada frase. Muestra barra de progreso simple.
        Devuelve lista de arrays numpy (float, 24000 Hz).
        """
        audio_chunks = []
        total = len(sentences)

        for i, sentence in enumerate(sentences, start=1):
            self._print_progress(i, total, sentence)

            out = self._model.inference(
                text=sentence,
                language=self.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
                speed=self.speed,
                num_beams=DEFAULT_NUM_GPT_OUTPUTS,
                enable_text_splitting=False,
            )

            audio_chunks.append(out["wav"])

        return audio_chunks

    # ------------------------------------------------------------------
    # Guardado del audio
    # ------------------------------------------------------------------

    def _save_audio(self, audio_chunks: list, output_path: Optional[str]) -> Path:
        """Concatena los chunks y guarda el WAV final."""
        import soundfile as sf
        import numpy as np

        # Se inserta un silencio de 150 ms entre frases para simular pausas naturales.
        # Sin este silencio, la concatenación de chunks suena entrecortada.
        silence = np.zeros(int(OUTPUT_SAMPLE_RATE * 0.15), dtype=np.float32)
        combined = []
        for i, chunk in enumerate(audio_chunks):
            arr = np.array(chunk, dtype=np.float32)
            combined.append(arr)
            if i < len(audio_chunks) - 1:
                combined.append(silence)

        audio = np.concatenate(combined)

        # Normalizar al 95% del rango para evitar clipping sin silenciar el audio.
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95

        # Resolver ruta de salida
        if output_path is None:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path.cwd() / f"{self.voice_name}_{ts}.wav"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), audio, OUTPUT_SAMPLE_RATE, subtype="PCM_16")
        return output_path.resolve()

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _read_text(self, source: str) -> str:
        """Lee texto desde un archivo .txt o lo devuelve tal cual si es string."""
        path = Path(source)
        if path.exists() and path.suffix.lower() == ".txt":
            return path.read_text(encoding="utf-8")
        # Tratar como texto directo
        return source

    def _print_progress(self, current: int, total: int, sentence: str) -> None:
        preview = sentence[:55] + "..." if len(sentence) > 55 else sentence
        bar_len = 20
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f'  [{bar}] {current}/{total}  "{preview}"', end="\r", flush=True)
        if current == total:
            print()  # salto de línea al terminar
