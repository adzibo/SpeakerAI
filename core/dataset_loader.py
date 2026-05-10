"""
core/dataset_loader.py
----------------------
Validación y carga del dataset antes del fine-tuning.

Verifica:
  - Existencia de la carpeta del dataset y sus archivos.
  - Formato correcto de metadata.csv (pipe-separated: filename|transcription).
  - Existencia de cada WAV referenciado en el CSV.
  - Parámetros de audio: 22050 Hz, mono, PCM 16-bit.
  - Duración total del dataset (mínimo recomendado: 30 min).

Uso:
    from core.dataset_loader import DatasetLoader
    loader = DatasetLoader("datasets/audioAdil")
    info = loader.validate()
"""

import csv
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# Parámetros de audio exigidos (los que se exportan desde Audacity)
REQUIRED_SAMPLE_RATE = 22050
REQUIRED_CHANNELS = 1   # mono
REQUIRED_SAMPWIDTH = 2  # PCM 16-bit → 2 bytes por muestra

MIN_DURATION_SECONDS = 30 * 60    # 30 minutos mínimo recomendado
MAX_DURATION_SECONDS = 120 * 60   # 120 minutos máximo (más no aporta)


@dataclass
class DatasetInfo:
    """
    Resultado de la validación de un dataset.
    Se devuelve siempre, aunque haya errores: el caller debe comprobar is_valid
    antes de continuar con el entrenamiento.
    """
    name: str
    path: Path
    wavs_dir: Path
    metadata_path: Path
    num_samples: int
    total_duration_s: float
    avg_duration_s: float
    errors: List[str]       # errores bloqueantes — impiden el entrenamiento
    warnings: List[str]     # advertencias — el entrenamiento puede continuar pero con riesgo

    @property
    def is_valid(self) -> bool:
        """Un dataset es válido si no tiene ningún error bloqueante."""
        return len(self.errors) == 0

    @property
    def total_duration_fmt(self) -> str:
        """Duración total formateada como HH:MM:SS para mostrar en consola."""
        h = int(self.total_duration_s // 3600)
        m = int((self.total_duration_s % 3600) // 60)
        s = int(self.total_duration_s % 60)
        return f"{h:02d}h {m:02d}m {s:02d}s"


class DatasetLoader:
    """Valida y describe un dataset en formato SpeakerTTS."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.wavs_dir = self.dataset_path / "wavs"
        self.metadata_path = self.dataset_path / "metadata.csv"

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def validate(self, verbose: bool = True) -> DatasetInfo:
        """
        Ejecuta todas las validaciones y devuelve un DatasetInfo.
        Si verbose=True, imprime el resumen por consola.
        """
        errors: List[str] = []
        warnings: List[str] = []
        samples: List[Tuple[str, str]] = []  # (filename, transcription)
        durations: List[float] = []

        # 1) Existencia de rutas base
        if not self.dataset_path.exists():
            errors.append(f"Carpeta de dataset no encontrada: {self.dataset_path}")
            return self._build_info(samples, durations, errors, warnings)

        if not self.wavs_dir.exists():
            errors.append(f"Carpeta 'wavs/' no encontrada en: {self.dataset_path}")

        if not self.metadata_path.exists():
            errors.append(f"'metadata.csv' no encontrado en: {self.dataset_path}")

        if errors:
            return self._build_info(samples, durations, errors, warnings)

        # 2) Parseo de metadata.csv
        samples, parse_errors = self._parse_metadata()
        errors.extend(parse_errors)
        if errors:
            return self._build_info(samples, durations, errors, warnings)

        if len(samples) == 0:
            errors.append("metadata.csv está vacío — no hay muestras.")
            return self._build_info(samples, durations, errors, warnings)

        # 3) Validación WAV por WAV
        for filename, _ in samples:
            wav_path = self.wavs_dir / filename
            if not wav_path.exists():
                errors.append(f"WAV no encontrado: {wav_path}")
                continue
            dur, wav_errors, wav_warnings = self._validate_wav(wav_path)
            errors.extend(wav_errors)
            warnings.extend(wav_warnings)
            if dur is not None:
                durations.append(dur)

        # 4) Duración total
        total = sum(durations)
        if total < MIN_DURATION_SECONDS:
            warnings.append(
                f"Duración total ({total/60:.1f} min) menor al mínimo recomendado "
                f"({MIN_DURATION_SECONDS//60} min). La calidad del fine-tuning puede ser baja."
            )
        if total > MAX_DURATION_SECONDS:
            warnings.append(
                f"Duración total ({total/60:.1f} min) supera los "
                f"{MAX_DURATION_SECONDS//60} min. Considera reducir el dataset."
            )

        info = self._build_info(samples, durations, errors, warnings)

        if verbose:
            self._print_report(info)

        return info

    def get_metadata_rows(self) -> List[Tuple[str, str]]:
        """Devuelve las filas del CSV sin validación de audio."""
        rows, _ = self._parse_metadata()
        return rows

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _parse_metadata(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Lee metadata.csv y devuelve la lista de muestras y los errores encontrados.
        El formato esperado es pipe-separated (|) sin cabecera:
            nombre_segmento.wav|Texto transcrito del segmento.
        """
        samples: List[Tuple[str, str]] = []
        errors: List[str] = []

        try:
            with open(self.metadata_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                for i, row in enumerate(reader, start=1):
                    if not row or all(cell.strip() == "" for cell in row):
                        continue  # línea vacía, ignorar
                    if len(row) < 2:
                        errors.append(
                            f"metadata.csv línea {i}: formato incorrecto "
                            f"(esperado 'filename.wav|transcripción'), obtenido: {row}"
                        )
                        continue
                    filename = row[0].strip()
                    transcription = row[1].strip()
                    if not filename.endswith(".wav"):
                        errors.append(
                            f"metadata.csv línea {i}: nombre de archivo sin extensión .wav: '{filename}'"
                        )
                    if not transcription:
                        errors.append(
                            f"metadata.csv línea {i}: transcripción vacía para '{filename}'"
                        )
                    samples.append((filename, transcription))
        except UnicodeDecodeError:
            errors.append("metadata.csv no está codificado en UTF-8.")
        except Exception as e:
            errors.append(f"Error leyendo metadata.csv: {e}")

        return samples, errors

    def _validate_wav(
        self, wav_path: Path
    ) -> Tuple[float | None, List[str], List[str]]:
        """
        Valida un WAV individual.
        Devuelve (duración_segundos, errores, advertencias).
        """
        errors: List[str] = []
        warnings: List[str] = []
        duration = None

        try:
            with wave.open(str(wav_path), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                nframes = wf.getnframes()

                duration = nframes / sr

                if sr != REQUIRED_SAMPLE_RATE:
                    errors.append(
                        f"{wav_path.name}: sample rate {sr} Hz "
                        f"(requerido: {REQUIRED_SAMPLE_RATE} Hz)"
                    )
                if ch != REQUIRED_CHANNELS:
                    errors.append(
                        f"{wav_path.name}: {ch} canales "
                        f"(requerido: mono = {REQUIRED_CHANNELS})"
                    )
                if sw != REQUIRED_SAMPWIDTH:
                    errors.append(
                        f"{wav_path.name}: sample width {sw*8}-bit "
                        f"(requerido: {REQUIRED_SAMPWIDTH*8}-bit PCM)"
                    )

                # Segmentos muy cortos o muy largos
                if duration < 1.0:
                    warnings.append(
                        f"{wav_path.name}: duración muy corta ({duration:.2f}s). "
                        "Puede degradar el entrenamiento."
                    )
                if duration > 30.0:
                    warnings.append(
                        f"{wav_path.name}: duración muy larga ({duration:.1f}s). "
                        "XTTS tiene contexto máximo ~30s."
                    )

        except wave.Error as e:
            errors.append(f"{wav_path.name}: archivo WAV corrupto o inválido — {e}")
        except Exception as e:
            errors.append(f"{wav_path.name}: error inesperado — {e}")

        return duration, errors, warnings

    def _build_info(
        self,
        samples: List[Tuple[str, str]],
        durations: List[float],
        errors: List[str],
        warnings: List[str],
    ) -> DatasetInfo:
        total = sum(durations)
        avg = total / len(durations) if durations else 0.0
        return DatasetInfo(
            name=self.dataset_path.name,
            path=self.dataset_path,
            wavs_dir=self.wavs_dir,
            metadata_path=self.metadata_path,
            num_samples=len(samples),
            total_duration_s=total,
            avg_duration_s=avg,
            errors=errors,
            warnings=warnings,
        )

    def _print_report(self, info: DatasetInfo) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  VALIDACIÓN DEL DATASET: {info.name}")
        print(sep)
        print(f"  Ruta         : {info.path}")
        print(f"  Muestras     : {info.num_samples}")
        print(f"  Duración     : {info.total_duration_fmt}")
        print(f"  Duración avg : {info.avg_duration_s:.2f}s por segmento")

        if info.warnings:
            print(f"\n  ⚠  ADVERTENCIAS ({len(info.warnings)}):")
            for w in info.warnings:
                print(f"     • {w}")

        if info.errors:
            print(f"\n  ✗  ERRORES ({len(info.errors)}):")
            for e in info.errors:
                print(f"     • {e}")
            print(f"\n  Estado: ✗ DATASET INVÁLIDO — corrige los errores antes de entrenar.")
        else:
            print(f"\n  Estado: ✓ DATASET VÁLIDO")

        print(sep + "\n")
