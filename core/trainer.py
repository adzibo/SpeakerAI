"""
core/trainer.py
---------------
Orquesta el fine-tuning del GPT encoder de XTTS v2.

Pipeline correcto (según recipes/ljspeech/xtts_v2/train_gpt_xtts.py oficial):
  1. Valida el dataset con DatasetLoader.
  2. Descarga dvae.pth y mel_stats.pth desde HuggingFace si no están en base/.
  3. Construye GPTArgs, XttsAudioConfig y GPTTrainerConfig programáticamente.
  4. Carga las muestras con load_tts_samples usando el formatter 'coqui'.
  5. Instancia GPTTrainer y lanza Trainer(...).fit().
  6. Al terminar, copia el best_model.pth a models/trained/<name>/.
  7. Calcula y persiste el speaker_embedding.

Imports correctos de Coqui TTS:
  from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
  from TTS.tts.datasets import load_tts_samples
  from TTS.config.shared_configs import BaseDatasetConfig
  from trainer import Trainer, TrainerArgs
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from core.dataset_loader import DatasetLoader
from core.model_manager import ModelManager


# URLs de los ficheros auxiliares requeridos para el entrenamiento
DVAE_CHECKPOINT_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DEFAULT_LANGUAGE = "es"

# Hiperparámetros optimizados para RTX 4080 SUPER (16 GB VRAM)
# BATCH_SIZE * GRAD_ACUMM_STEPS >= 252 recomendado por Coqui
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 63  # 4 * 63 = 252
DEFAULT_EPOCHS = 10
DEFAULT_LR = 5e-6


class Trainer:
    """
    Orquesta el fine-tuning del GPT encoder de XTTS v2 para una voz concreta.
    Solo se fine-tunea el GPT encoder; el decoder DVAE y el vocoder permanecen
    congelados y se reutilizan desde models/base/.
    """

    def __init__(
        self,
        dataset_name: str,
        voice_name: str,
        language: str = DEFAULT_LANGUAGE,
        project_root: str = ".",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.voice_name = voice_name
        self.language = language
        self.epochs = epochs if epochs is not None else DEFAULT_EPOCHS
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE

        self.root = Path(project_root).resolve()
        self.mm = ModelManager(project_root)
        self.dataset_path = self.root / "datasets" / dataset_name

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Ejecuta el pipeline completo de fine-tuning."""
        print(f"\n{'='*60}")
        print(f"  SpeakerTTS — Fine-tuning")
        print(f"  Dataset : {self.dataset_name}")
        print(f"  Voz     : {self.voice_name}")
        print(f"  Idioma  : {self.language}")
        print(f"  Epochs  : {self.epochs}")
        print(f"  Batch   : {self.batch_size}  (grad_accum: {DEFAULT_GRAD_ACCUM})")
        print(f"{'='*60}\n")

        # 1) Validar dataset
        self._validate_dataset()

        # 2) Verificar modelo base
        if not self.mm.verify_base_model():
            raise RuntimeError(
                "Modelo base incompleto. Ejecuta: python3 speaker.py --download-base"
            )

        # 3) Preparar directorios
        training_dir = self.mm.prepare_training_dir(self.voice_name)
        trained_dir = self.mm.prepare_trained_dir(self.voice_name)

        # 4) Obtener dvae.pth y mel_stats.pth (requeridos para tokenización en training)
        dvae_path, mel_norm_path = self._ensure_dvae_files(training_dir)

        # 5) Lanzar entrenamiento
        out_path = training_dir / "checkpoints"
        self._launch_training(out_path, dvae_path, mel_norm_path)

        # 6) Exportar modelo final
        self._export_model(out_path, trained_dir)

        # 7) Calcular y cachear speaker embedding
        self._cache_speaker_embedding(trained_dir)

        print(f"\n{'='*60}")
        print(f"  ✓ Fine-tuning completado.")
        print(f"  Voz '{self.voice_name}' disponible en: {trained_dir}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Pasos privados
    # ------------------------------------------------------------------

    def _validate_dataset(self) -> None:
        loader = DatasetLoader(str(self.dataset_path))
        info = loader.validate(verbose=True)
        if not info.is_valid:
            raise ValueError(
                f"Dataset '{self.dataset_name}' no válido. "
                "Corrige los errores antes de entrenar."
            )

    def _ensure_dvae_files(self, training_dir: Path):
        """
        Garantiza que dvae.pth y mel_stats.pth están disponibles en training/<voz>/xtts_base_files/.
        Se intenta copiar dvae.pth desde models/base/ (evita descarga si ya existe).
        Si alguno falta, se descarga desde los servidores de Coqui.
        Estos ficheros son necesarios para tokenizar mel spectrograms durante el entrenamiento.
        """
        aux_dir = training_dir / "xtts_base_files"
        aux_dir.mkdir(parents=True, exist_ok=True)

        dvae_path = aux_dir / "dvae.pth"
        mel_norm_path = aux_dir / "mel_stats.pth"

        # Intentar copiar dvae desde models/base/ primero
        base_dvae = self.mm.get_base_dir() / "dvae.pth"
        if base_dvae.exists() and not dvae_path.exists():
            shutil.copy2(base_dvae, dvae_path)
            print(f"  ✓ dvae.pth copiado desde models/base/")

        # Descargar los que aún falten
        to_download = []
        if not dvae_path.exists():
            to_download.append((DVAE_CHECKPOINT_URL, "dvae.pth"))
        if not mel_norm_path.exists():
            to_download.append((MEL_NORM_URL, "mel_stats.pth"))

        if to_download:
            print(f"  Descargando ficheros auxiliares de entrenamiento...")
            try:
                from TTS.utils.manage import ModelManager as CoquiMM
                urls = [u for u, _ in to_download]
                CoquiMM._download_model_files(urls, str(aux_dir), progress_bar=True)
                print(f"  ✓ Descargados en: {aux_dir}")
            except Exception as e:
                raise RuntimeError(
                    f"Error descargando ficheros auxiliares: {e}\n"
                    f"Descárgalos manualmente en {aux_dir}:\n"
                    f"  {DVAE_CHECKPOINT_URL}\n"
                    f"  {MEL_NORM_URL}"
                ) from e

        return dvae_path, mel_norm_path

    def _launch_training(
        self,
        out_path: Path,
        dvae_path: Path,
        mel_norm_path: Path,
    ) -> None:
        """
        Construye la configuración programáticamente e instancia
        GPTTrainer + Trainer de Coqui TTS.
        """
        print("  Cargando Coqui TTS...", end=" ", flush=True)

        try:
            from TTS.tts.layers.xtts.trainer.gpt_trainer import (
                GPTArgs,
                GPTTrainer,
                GPTTrainerConfig,
            )
            from TTS.tts.models.xtts import XttsAudioConfig
            from TTS.tts.datasets import load_tts_samples
            from TTS.config.shared_configs import BaseDatasetConfig
            from trainer import Trainer as CoquiTrainer, TrainerArgs
        except ImportError as e:
            raise ImportError(
                f"Error importando Coqui TTS: {e}\n"
                "Asegúrate de que coqui-tts está instalado: pip install coqui-tts"
            ) from e

        print("OK")

        base_dir = self.mm.get_base_dir()
        xtts_checkpoint = str(base_dir / "model.pth")
        tokenizer_file = str(base_dir / "vocab.json")

        # ── GPTArgs ────────────────────────────────────────────────────
        # Parámetros de arquitectura del GPT encoder de XTTS v2.
        # Deben coincidir con los del modelo base; no cambiar salvo que se
        # actualice la versión del modelo base.
        model_args = GPTArgs(
            max_conditioning_length=132300,     # 6 s × 22050 Hz = máx. audio de referencia
            min_conditioning_length=66150,      # 3 s × 22050 Hz = mín. audio de referencia
            debug_loading_failures=False,
            max_wav_length=255995,              # ~11.6 s — longitud máx. de segmento de entrenamiento
            max_text_length=200,                # máx. caracteres por segmento de texto
            mel_norm_file=str(mel_norm_path),
            dvae_checkpoint=str(dvae_path),
            xtts_checkpoint=xtts_checkpoint,    # pesos de partida para el fine-tuning
            tokenizer_file=tokenizer_file,
            gpt_num_audio_tokens=1026,          # tokens de audio del vocabulario del GPT
            gpt_start_audio_token=1024,         # token especial de inicio de secuencia de audio
            gpt_stop_audio_token=1025,          # token especial de fin de secuencia de audio
            gpt_use_masking_gt_prompt_approach=True,    # técnica de enmascaramiento del prompt
            gpt_use_perceiver_resampler=True,           # resampler para el conditioning
        )

        # ── Audio config ───────────────────────────────────────────────
        audio_config = XttsAudioConfig(
            sample_rate=22050,
            dvae_sample_rate=22050,
            output_sample_rate=24000,
        )

        # ── WAV de referencia para frases de test en TensorBoard ───────
        ref_wavs = sorted((self.dataset_path / "wavs").glob("*.wav"))
        speaker_ref = [str(ref_wavs[0])] if ref_wavs else []

        # ── GPTTrainerConfig ───────────────────────────────────────────
        run_name = f"GPT_XTTS_{self.voice_name}_FT"

        config = GPTTrainerConfig(
            epochs=self.epochs,
            output_path=str(out_path),
            model_args=model_args,
            run_name=run_name,
            project_name="SpeakerTTS",
            run_description=f"SpeakerTTS fine-tuning — voz: {self.voice_name}",
            dashboard_logger="tensorboard",
            logger_uri=None,
            audio=audio_config,
            batch_size=self.batch_size,
            batch_group_size=48,
            eval_batch_size=max(1, self.batch_size // 2),
            num_loader_workers=8,
            eval_split_max_size=256,
            print_step=50,
            plot_step=100,
            log_model_step=1000,
            save_step=10000,
            save_n_checkpoints=1,
            save_checkpoints=True,
            optimizer="AdamW",
            optimizer_wd_only_on_weights=True,
            optimizer_params={
                "betas": [0.9, 0.96],
                "eps": 1e-8,
                "weight_decay": 1e-2,
            },
            lr=DEFAULT_LR,
            lr_scheduler="MultiStepLR",
            lr_scheduler_params={
                "milestones": [50000, 150000, 300000],
                "gamma": 0.5,
                "last_epoch": -1,
            },
            test_sentences=[
                {
                    "text": "Hola, esta es una prueba de síntesis de voz con mi modelo entrenado.",
                    "speaker_wav": speaker_ref,
                    "language": self.language,
                },
            ],
        )

        # ── Formatter propio para el formato SpeakerTTS ────────────────
        # Formato: filename.wav|transcripción  (sin cabecera, pipe-separated)
        # El formatter coqui nativo exige cabecera con columnas audio_file|text,
        # por lo que usamos un formatter personalizado registrado en load_tts_samples.
        import os as _os

        dataset_path_str = str(self.dataset_path)
        dataset_name_str = self.dataset_name
        language_str = self.language

        def speakertts_formatter(root_path, meta_file, **kwargs):
            """
            Lee metadata.csv en formato SpeakerTTS:
              audioNombre_0001.wav|Transcripción del segmento.
            Devuelve lista de dicts con las claves requeridas por Coqui TTS.
            """
            items = []
            meta_path = _os.path.join(root_path, meta_file)
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|", 1)
                    if len(parts) < 2:
                        continue
                    filename, text = parts[0].strip(), parts[1].strip()
                    audio_file = _os.path.join(root_path, "wavs", filename)
                    items.append({
                        "audio_file":   audio_file,
                        "text":         text,
                        "speaker_name": dataset_name_str,
                        "language":     language_str,
                        "root_path":    root_path,
                    })
            return items

        # ── Dataset config ─────────────────────────────────────────────
        dataset_config = BaseDatasetConfig(
            formatter="speakertts",
            dataset_name=self.dataset_name,
            path=str(self.dataset_path),
            meta_file_train="metadata.csv",
            language=self.language,
        )

        # ── Cargar muestras ────────────────────────────────────────────
        print("  Cargando muestras del dataset...")

        try:
            from TTS.tts.datasets import register_formatter
            register_formatter("speakertts", speakertts_formatter)
            train_samples, eval_samples = load_tts_samples(
                [dataset_config],
                eval_split=True,
                eval_split_max_size=config.eval_split_max_size,
                eval_split_size=0.01,
            )
        except (ImportError, TypeError):
            # Versiones antiguas sin register_formatter: pasar formatter directamente
            train_samples, eval_samples = load_tts_samples(
                [dataset_config],
                eval_split=True,
                eval_split_max_size=config.eval_split_max_size,
                eval_split_size=0.01,
                formatter=speakertts_formatter,
            )

        print(f"  ✓ {len(train_samples)} train / {len(eval_samples)} eval")

        # ── Instanciar GPTTrainer ──────────────────────────────────────
        model = GPTTrainer.init_from_config(config)

        # Si el usuario cambió el batch_size, recalcular grad_accum para que
        # batch × accum >= 252 (mínimo recomendado por Coqui).
        # Se usa ceil division: -(-a // b) es equivalente a math.ceil(a / b) sin importar math.
        target_product = 252
        grad_accum = DEFAULT_GRAD_ACCUM
        if self.batch_size * grad_accum < target_product:
            grad_accum = -(-target_product // self.batch_size)  # ceil division

        print(f"\n  Iniciando entrenamiento...")
        print(f"  grad_accum_steps: {grad_accum}  (batch {self.batch_size} × {grad_accum} = {self.batch_size * grad_accum})")
        print(f"  TensorBoard: tensorboard --logdir {out_path}\n")

        trainer = CoquiTrainer(
            TrainerArgs(
                restore_path=None,
                skip_train_epoch=False,
                start_with_eval=True,
                grad_accum_steps=grad_accum,
            ),
            config,
            output_path=str(out_path),
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        trainer.fit()

    def _export_model(self, out_path: Path, trained_dir: Path) -> None:
        """
        Copia los ficheros del mejor checkpoint a models/trained/<nombre>/.
        Coqui TTS genera best_model.pth cuando el modelo mejora en validación.
        Si no existe (entrenamiento interrumpido), se usa el último .pth disponible.
        """
        print(f"\n  Exportando modelo final a: {trained_dir}")

        # best_model.pth es el checkpoint con mejor pérdida de validación.
        candidates = sorted(out_path.rglob("best_model.pth"))
        if not candidates:
            # Fallback: último checkpoint guardado si no hay best_model.
            candidates = sorted(out_path.rglob("*.pth"))

        if not candidates:
            raise FileNotFoundError(
                f"No se encontró ningún checkpoint en: {out_path}"
            )

        best = candidates[-1]
        shutil.copy2(best, trained_dir / "model.pth")
        print(f"  ✓ model.pth  ←  {best.name}")

        # config.json del run
        config_src = best.parent / "config.json"
        if config_src.exists():
            shutil.copy2(config_src, trained_dir / "config.json")
            print(f"  ✓ config.json")

        # Se copia vocab.json desde el modelo base para que la carpeta de la voz
        # sea autónoma y no dependa de models/base/ en inferencia.
        base_vocab = self.mm.get_base_dir() / "vocab.json"
        if base_vocab.exists():
            shutil.copy2(base_vocab, trained_dir / "vocab.json")
            print(f"  ✓ vocab.json")

    def _cache_speaker_embedding(self, trained_dir: Path) -> None:
        """
        Calcula el speaker embedding usando los WAVs del dataset y lo guarda en disco.
        Hacerlo aquí, justo tras el entrenamiento, evita tener que recalcularlo
        en la primera ejecución de generate(). Si falla, la generación lo calculará
        igualmente al vuelo, así que no es un error bloqueante.
        """
        print(f"\n  Calculando speaker embedding...")

        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError:
            print("  ⚠ No se pudo calcular el embedding.")
            return

        config_path = trained_dir / "config.json"
        model_path = trained_dir / "model.pth"
        vocab_path = trained_dir / "vocab.json"

        if not all(p.exists() for p in [config_path, model_path, vocab_path]):
            print("  ⚠ Ficheros del modelo incompletos, embedding no calculado.")
            return

        ref_wavs = sorted((self.dataset_path / "wavs").glob("*.wav"))[:5]
        if not ref_wavs:
            print("  ⚠ No se encontraron WAVs de referencia.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = XttsConfig()
        config.load_json(str(config_path))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(model_path),
            vocab_path=str(vocab_path),
            eval=True,
        )
        model.to(device)

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(w) for w in ref_wavs]
        )

        self.mm.save_speaker_embedding(
            self.voice_name, gpt_cond_latent, speaker_embedding
        )
        print(f"  ✓ Speaker embedding calculado con {len(ref_wavs)} WAVs de referencia.")
