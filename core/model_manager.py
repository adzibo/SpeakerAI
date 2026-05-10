"""
core/model_manager.py
---------------------
Gestión del ciclo de vida de los modelos entrenados.

Responsabilidades:
  - Listar las voces fine-tuned disponibles.
  - Verificar la integridad de los ficheros de un modelo.
  - Eliminar completamente una voz (trained + training artifacts).
  - Resolver rutas del modelo base y de voces entrenadas.
  - Guardar el speaker_embedding una vez calculado.

Uso:
    from core.model_manager import ModelManager
    mm = ModelManager()
    mm.list_voices()
    mm.delete_voice("Adil")
"""

import json
import shutil
import torch
from pathlib import Path
from typing import Dict, List, Optional


# Ficheros mínimos para que una voz entrenada sea usable en inferencia.
# Si falta alguno, la voz se marca como inválida y no se puede usar para generar audio.
REQUIRED_TRAINED_FILES = [
    "model.pth",    # pesos del GPT encoder fine-tuneado
    "config.json",  # configuración del modelo (arquitectura y parámetros)
    "vocab.json",   # vocabulario del tokenizador
]

# Ficheros necesarios en models/base/ para el entrenamiento y la inferencia.
# dvae.pth está integrado en model.pth desde xtts_v2 y no se requiere como fichero aparte.
REQUIRED_BASE_FILES = [
    "model.pth",            # pesos del modelo base preentrenado
    "config.json",          # configuración base de XTTS v2
    "vocab.json",           # vocabulario compartido entre base y voces entrenadas
    "speakers_xtts.pth",    # embeddings de los speakers del modelo original
]


class ModelManager:
    """Gestiona rutas y ciclo de vida de los modelos de SpeakerTTS."""

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root).resolve()
        self.base_dir = self.root / "models" / "base"
        self.trained_dir = self.root / "models" / "trained"
        self.training_dir = self.root / "training"

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def list_voices(self, verbose: bool = True) -> List[Dict]:
        """
        Lista todas las voces fine-tuned disponibles.
        Devuelve lista de dicts con metadata de cada voz.
        """
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        voices = []

        for d in sorted(self.trained_dir.iterdir()):
            if not d.is_dir():
                continue
            missing = self._missing_files(d, REQUIRED_TRAINED_FILES)
            has_embedding = (d / "speaker_embedding.pth").exists()
            info = {
                "name": d.name,
                "path": str(d),
                "valid": len(missing) == 0,
                "missing_files": missing,
                "has_embedding": has_embedding,
            }
            voices.append(info)

        if verbose:
            self._print_voices(voices)

        return voices

    def voice_exists(self, name: str) -> bool:
        return (self.trained_dir / name).exists()

    def get_voice_path(self, name: str) -> Path:
        return self.trained_dir / name

    def get_training_path(self, name: str) -> Path:
        return self.training_dir / name

    def is_voice_valid(self, name: str) -> bool:
        voice_dir = self.trained_dir / name
        if not voice_dir.exists():
            return False
        return len(self._missing_files(voice_dir, REQUIRED_TRAINED_FILES)) == 0

    def verify_base_model(self) -> bool:
        """Comprueba que el modelo base xtts_v2 está completo."""
        missing = self._missing_files(self.base_dir, REQUIRED_BASE_FILES)
        if missing:
            print(f"\n  ✗ Modelo base incompleto. Faltan: {missing}")
            print(f"    Ejecuta: python3 speaker.py --download-base")
            return False
        return True

    # ------------------------------------------------------------------
    # Rutas útiles para trainer y synthesizer
    # ------------------------------------------------------------------

    def get_base_dir(self) -> Path:
        return self.base_dir

    def prepare_trained_dir(self, name: str) -> Path:
        """Crea y devuelve la carpeta del modelo entrenado."""
        path = self.trained_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare_training_dir(self, name: str) -> Path:
        """Crea y devuelve la carpeta de artefactos de entrenamiento."""
        path = self.training_dir / name
        (path / "logs").mkdir(parents=True, exist_ok=True)
        (path / "checkpoints").mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Speaker Embedding
    # ------------------------------------------------------------------

    def save_speaker_embedding(
        self,
        name: str,
        gpt_cond_latent: "torch.Tensor",
        speaker_embedding: "torch.Tensor",
    ) -> None:
        """
        Persiste los latents del speaker en disco (speaker_embedding.pth).
        Calcularlos desde los WAVs tarda varios segundos; guardarlos evita repetir
        ese cálculo en cada generación de audio.
        """
        voice_dir = self.trained_dir / name
        voice_dir.mkdir(parents=True, exist_ok=True)
        emb_path = voice_dir / "speaker_embedding.pth"
        torch.save(
            {
                "gpt_cond_latent":  gpt_cond_latent,
                "speaker_embedding": speaker_embedding,
            },
            emb_path,
        )
        print(f"  ✓ Speaker embedding guardado: {emb_path}")

    def load_speaker_embedding(self, name: str) -> Optional[Dict]:
        """
        Carga el speaker embedding cacheado. Devuelve None si no existe.
        Se usa map_location="cpu" para compatibilidad: el caller mueve los tensores
        al dispositivo correcto (GPU/CPU) después de cargarlos.
        """
        emb_path = self.trained_dir / name / "speaker_embedding.pth"
        if not emb_path.exists():
            return None
        return torch.load(emb_path, map_location="cpu")

    # ------------------------------------------------------------------
    # Eliminación
    # ------------------------------------------------------------------

    def delete_voice(self, name: str, force: bool = False) -> bool:
        """
        Elimina completamente una voz entrenada:
          - models/trained/<name>/
          - training/<name>/

        Si force=False, pide confirmación por consola.
        Devuelve True si se eliminó, False si se canceló.
        """
        trained_path = self.trained_dir / name
        training_path = self.training_dir / name

        if not trained_path.exists() and not training_path.exists():
            print(f"\n  ✗ La voz '{name}' no existe.")
            return False

        if not force:
            print(f"\n  ⚠  Vas a eliminar PERMANENTEMENTE la voz '{name}'.")
            print(f"     Se borrarán:")
            if trained_path.exists():
                print(f"       • {trained_path}")
            if training_path.exists():
                print(f"       • {training_path}")
            confirm = input("     Confirmar (escribe el nombre de la voz para confirmar): ").strip()
            if confirm != name:
                print("  Eliminación cancelada.")
                return False

        if trained_path.exists():
            shutil.rmtree(trained_path)
            print(f"  ✓ Eliminado: {trained_path}")

        if training_path.exists():
            shutil.rmtree(training_path)
            print(f"  ✓ Eliminado: {training_path}")

        print(f"  ✓ Voz '{name}' eliminada completamente.\n")
        return True

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _missing_files(self, directory: Path, required: List[str]) -> List[str]:
        """Devuelve la lista de ficheros requeridos que no están presentes en el directorio."""
        return [f for f in required if not (directory / f).exists()]

    def _print_voices(self, voices: List[Dict]) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print("  VOCES ENTRENADAS DISPONIBLES")
        print(sep)
        if not voices:
            print("  (ninguna — entrena una voz con: python3 speaker.py -t <dataset> -n <nombre>)")
        else:
            for v in voices:
                status = "✓" if v["valid"] else "✗"
                emb = "embedding ✓" if v["has_embedding"] else "embedding ✗ (se calculará en inferencia)"
                print(f"  {status}  {v['name']:<20} {emb}")
                if v["missing_files"]:
                    print(f"      Faltan: {v['missing_files']}")
        print(sep + "\n")
