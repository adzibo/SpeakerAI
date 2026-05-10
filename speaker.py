"""
speaker.py
----------
CLI principal de SpeakerTTS.

Comandos disponibles:
  Entrenamiento:
    python3 speaker.py -t <dataset> -n <nombre> [opciones]

  Generación:
    python3 speaker.py -g <nombre_voz> <archivo.txt> [-o salida.wav]

  Gestión:
    python3 speaker.py --list
    python3 speaker.py --delete <nombre_voz>
    python3 speaker.py --download-base

Ejemplos:
    python3 speaker.py -t audioAdil -n VozAdil
    python3 speaker.py -g VozAdil descripcion.txt -o VoiceFile.wav
    python3 speaker.py --list
    python3 speaker.py --delete VozAdil
"""

import argparse
import sys
from pathlib import Path


# ── Raíz del proyecto = directorio donde reside speaker.py ─────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent)

# ── BANNER ─────────────────────────────────────────────────────────────────
BANNER = r"""
  ███████╗██████╗ ███████╗ █████╗ ██╗  ██╗███████╗██████╗      █████╗ ██╗
  ██╔════╝██╔══██╗██╔════╝██╔══██╗██║ ██╔╝██╔════╝██╔══██╗    ██╔══██╗██║
  ███████╗██████╔╝█████╗  ███████║█████╔╝ █████╗  ██████╔╝    ███████║██║
  ╚════██║██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗ ██╔══╝  ██╔══██╗    ██╔══██║██║
  ███████║██║     ███████╗██║  ██║██║  ██╗███████╗██║  ██║    ██║  ██║██║
  ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
                                                       Created by AdZiBo
"""


# ══════════════════════════════════════════════════════════════════════════════
# PARSER
# ══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    """
        Define todos los argumentos aceptados por el CLI.
        Cada grupo de argumentos corresponde a un modo de uso:
          - Entrenamiento (-t / -n): lanza el fine-tuning de XTTS v2.
          - Generación (-g): sintetiza audio con una voz ya entrenada.
          - Gestión (--list, --delete, --download-base): administra modelos.
        """
    parser = argparse.ArgumentParser(
        prog="speaker.py",
        description="SpeakerTTS — Clonación y síntesis de voz local con XTTS v2",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Ejemplos:
          python3 speaker.py -t audioAdil -n Adil
          python3 speaker.py -t audioAdil -n Adil --lang es --epochs 10 --batch 8
          python3 speaker.py -g Adil descripcion.txt -o AdilVoiceFile.wav
          python3 speaker.py --list
          python3 speaker.py --delete Adil
          python3 speaker.py --download-base
                """,
    )

    # ── Modo entrenamiento ──────────────────────────────────────────────────
    train_group = parser.add_argument_group("Entrenamiento (fine-tuning)")
    train_group.add_argument(
        "-t", "--train",
        metavar="DATASET",
        help="Nombre del dataset a usar (carpeta en datasets/).\n"
             "Ejemplo: -t audioAdil",
    )
    train_group.add_argument(
        "-n", "--name",
        metavar="NOMBRE",
        help="Nombre identificador de la voz entrenada.\n"
             "Ejemplo: -n Adil",
    )
    train_group.add_argument(
        "--lang",
        metavar="IDIOMA",
        default="es",
        help="Código de idioma para el entrenamiento (default: es).\n"
             "Códigos válidos: es, en, fr, de, it, pt, nl, ru, ja, zh-cn, ko...",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        metavar="N",
        default=None,
        help="Número de épocas de entrenamiento.\n"
             "Sobreescribe el valor de la plantilla de configuración.",
    )
    train_group.add_argument(
        "--batch",
        type=int,
        metavar="N",
        default=None,
        help="Batch size.\n"
             "Sobreescribe el valor de la plantilla de configuración.",
    )

    # ── Modo generación ─────────────────────────────────────────────────────
    gen_group = parser.add_argument_group("Generación de audio")
    gen_group.add_argument(
        "-g", "--generate",
        metavar="NOMBRE_VOZ",
        help="Nombre de la voz fine-tuned a usar para la síntesis.\n"
             "Ejemplo: -g Adil",
    )
    gen_group.add_argument(
        "text_source",
        nargs="?",
        metavar="ARCHIVO.txt",
        help="Archivo de texto a sintetizar (requerido con -g).",
    )
    gen_group.add_argument(
        "-o", "--output",
        metavar="SALIDA.wav",
        default=None,
        help="Nombre del archivo WAV de salida.\n"
             "Por defecto: <Voz>_<timestamp>.wav en el directorio actual.",
    )
    gen_group.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="VELOCIDAD",
        help="Velocidad del habla (0.5–2.0, default: 1.0).",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        metavar="TEMP",
        help="Temperatura del GPT (0.1–1.0, default: 0.75).\n"
             "Valores bajos → más estable. Valores altos → más expresivo.",
    )

    # ── Gestión de modelos ──────────────────────────────────────────────────
    mgmt_group = parser.add_argument_group("Gestión de modelos")
    mgmt_group.add_argument(
        "--list",
        action="store_true",
        help="Lista todas las voces fine-tuned disponibles.",
    )
    mgmt_group.add_argument(
        "--delete",
        metavar="NOMBRE_VOZ",
        help="Elimina completamente una voz entrenada.\n"
             "Borra models/trained/<nombre>/ y training/<nombre>/.",
    )
    mgmt_group.add_argument(
        "--delete-force",
        metavar="NOMBRE_VOZ",
        help="Igual que --delete pero sin pedir confirmación.",
    )
    mgmt_group.add_argument(
        "--download-base",
        action="store_true",
        help="Descarga el modelo base xtts_v2 en models/base/.",
    )

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
def handle_train(args: argparse.Namespace) -> None:
    """
    Punto de entrada para el modo entrenamiento.
    Valida que se hayan proporcionado dataset (-t) y nombre de voz (-n),
    luego delega en Trainer.run() el pipeline completo de fine-tuning.
    """
    if not args.name:
        print("  ✗ Error: --name / -n es obligatorio con --train / -t")
        print("  Uso: python3 speaker.py -t <dataset> -n <nombre>")
        sys.exit(1)

    from core.trainer import Trainer

    trainer = Trainer(
        dataset_name=args.train,
        voice_name=args.name,
        language=args.lang,
        project_root=PROJECT_ROOT,
        epochs=args.epochs,
        batch_size=args.batch,
    )
    trainer.run()


def handle_generate(args: argparse.Namespace) -> None:
    """
    Punto de entrada para el modo generación.
    Acepta como fuente de texto:
      - Un archivo .txt (si la ruta existe en disco).
      - Un string corto pasado directamente por consola.
    Valida velocidad y delega en Synthesizer.generate().
    """
    if not args.text_source:
        print("  ✗ Error: debes indicar un archivo de texto para generar audio.")
        print("  Uso: python3 speaker.py -g <voz> <archivo.txt> [-o salida.wav]")
        sys.exit(1)

    if not Path(args.text_source).exists():
        # Tratar como texto directo si no es una ruta existente
        if len(args.text_source) < 5:
            print(f"  ✗ Error: archivo no encontrado: {args.text_source}")
            sys.exit(1)

    # Validar velocidad
    if not (0.5 <= args.speed <= 2.0):
        print(f"  ✗ Error: --speed debe estar entre 0.5 y 2.0 (dado: {args.speed})")
        sys.exit(1)

    from core.synthesizer import Synthesizer

    synth = Synthesizer(
        voice_name=args.generate,
        project_root=PROJECT_ROOT,
        temperature=args.temperature,
        speed=args.speed,
    )
    synth.generate(
        text_source=args.text_source,
        output_path=args.output,
    )


def handle_list() -> None:
    """Muestra por consola todas las voces entrenadas disponibles y su estado."""
    from core.model_manager import ModelManager
    mm = ModelManager(PROJECT_ROOT)
    mm.list_voices(verbose=True)


def handle_delete(name: str, force: bool = False) -> None:
    """
    Elimina una voz entrenada.
    Con force=False (por defecto) pide confirmación escribiendo el nombre de la voz.
    Con force=True (--delete-force) borra sin preguntar.
    """
    from core.model_manager import ModelManager
    mm = ModelManager(PROJECT_ROOT)
    mm.delete_voice(name, force=force)


def handle_download_base() -> None:
    """Descarga el modelo base xtts_v2 desde Hugging Face en models/base/."""
    base_dir = Path(PROJECT_ROOT) / "models" / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  SpeakerTTS — Descarga del modelo base xtts_v2")
    print(f"{'='*60}")
    print(f"  Destino: {base_dir}\n")

    try:
        import shutil
        from TTS.utils.manage import ModelManager as CoquiModelManager

        print("  Iniciando descarga (requiere conexión a internet)...")
        print("  Esto puede tardar varios minutos según la conexión.\n")

        manager = CoquiModelManager()
        model_path, config_path, _ = manager.download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        )

        # model_path puede ser:
        #   (a) ruta al fichero .pth  → su carpeta es .parent
        #   (b) ruta a la carpeta     → usarla directamente
        resolved = Path(model_path)
        src_dir = resolved.parent if resolved.is_file() else resolved

        # Verificación de seguridad: buscar model.pth dentro de src_dir
        if not (src_dir / "model.pth").exists():
            # Intentar con el directorio canónico del cache de Coqui
            cache_candidate = (
                Path.home()
                / ".local/share/tts"
                / "tts_models--multilingual--multi-dataset--xtts_v2"
            )
            if (cache_candidate / "model.pth").exists():
                src_dir = cache_candidate
            else:
                raise FileNotFoundError(
                    f"No se encontró model.pth en: {src_dir}\n"
                    f"  ni en: {cache_candidate}"
                )

        print(f"  Fuente: {src_dir}\n")

        # dvae.pth es opcional en xtts_v2 (integrado en model.pth desde v2)
        files_to_copy = [
            "model.pth",
            "config.json",
            "vocab.json",
            "speakers_xtts.pth",
            "dvae.pth",          # presente solo en algunas versiones
        ]

        copied = 0
        for fname in files_to_copy:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, base_dir / fname)
                print(f"  ✓ {fname}")
                copied += 1
            else:
                # dvae.pth ausente es normal en xtts_v2
                if fname != "dvae.pth":
                    print(f"  ✗ Fichero requerido no encontrado: {fname}")
                # Si falta un fichero requerido distinto de dvae, abortar
                if fname not in ("dvae.pth",):
                    raise FileNotFoundError(
                        f"Fichero requerido no encontrado en cache: {src}"
                    )

        print(f"\n  ✓ Modelo base listo en: {base_dir} ({copied} ficheros copiados)\n")

    except ImportError as e:
        print(f"  ✗ Coqui TTS no está instalado correctamente: {e}")
        print("  Instala con: pip install coqui-tts")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Error durante la descarga: {e}")
        print("\n  También puedes copiar manualmente los ficheros del cache:")
        cache = Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        print(f"  cp {cache}/*.pth {cache}/*.json {Path(PROJECT_ROOT) / 'models' / 'base'}/")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """
    Punto de entrada principal del CLI.
    Parsea los argumentos y enruta al handler correspondiente.
    El orden de evaluación determina la prioridad cuando se combinan flags.
    """
    print(BANNER)

    parser = build_parser()

    # Mostrar ayuda si no se pasan argumentos
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # ── Enrutado de comandos ────────────────────────────────────────────────

    if args.download_base:
        handle_download_base()

    elif args.list:
        handle_list()

    elif args.delete:
        handle_delete(args.delete, force=False)

    elif args.delete_force:
        handle_delete(args.delete_force, force=True)

    elif args.train:
        handle_train(args)

    elif args.generate:
        handle_generate(args)

    else:
        print("  ✗ No se reconoció ningún comando válido.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
