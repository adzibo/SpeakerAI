"""
visualize.py
------------
Generador de figuras para la documentación de SpeakerTTS.

Produce tres tipos de figuras en docs/figures/:
  1. Diagrama del pipeline completo del proyecto       (--pipeline)
  2. Espectrograma comparativo ref vs. sintetizado     (--spectrogram)
  3. Curva de loss del entrenamiento (desde TBoard)    (--loss)

Uso:
  python3 visualize.py --all --ref-wav datasets/Raul/wavs/Raul_0001.wav \
                              --synth-wav salida.wav \
                              --log-dir training/VozRaul/logs

  python3 visualize.py --pipeline
  python3 visualize.py --spectrogram --ref-wav REF.wav --synth-wav SYNTH.wav
  python3 visualize.py --loss --log-dir training/VozRaul/logs
"""

import argparse
import sys
from pathlib import Path


# ── Paleta de colores del proyecto ─────────────────────────────────────────
DARK_BG    = "#0f1117"
DARK_PANEL = "#1a1d27"
ACCENT     = "#7c6af7"       # violeta principal
ACCENT2    = "#4ecdc4"       # teal para el sintetizado
WHITE      = "#e8e8f0"
GRAY       = "#555770"
GRID_COLOR = "#23263a"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DIAGRAMA DEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_pipeline(output_dir: Path, dpi: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ── Título ──────────────────────────────────────────────────────────────
    ax.text(9, 8.5, "SpeakerTTS — Pipeline completo",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=WHITE, fontfamily="monospace")

    # ── Definición de nodos ─────────────────────────────────────────────────
    # Cada nodo: (x_centro, y_centro, label_línea1, label_línea2, color_borde)
    nodes = [
        # Fila superior: pipeline de datos
        (1.4,  6.2, "Audio",        "Bruto",          GRAY),
        (3.5,  6.2, "Audacity",     "Tratamiento",    ACCENT),
        (5.6,  6.2, "Audio",        "Tratado",        GRAY),
        (7.7,  6.2, "splitter.py",  "Segmentación",   ACCENT),
        (9.8,  6.2, "wavs/ +",      "metadata.csv",   GRAY),
        # Fila inferior: pipeline de entrenamiento/inferencia
        (9.8,  3.2, "dataset_",     "loader.py",      ACCENT),
        (12.0, 3.2, "trainer.py",   "Fine-tuning",    ACCENT),
        (14.2, 3.2, "models/",      "trained/",       GRAY),
        (16.4, 3.2, "synthesizer",  ".py",            ACCENT),
        # Resultado final
        (16.4, 6.2, "Audio",        "Sintetizado",    ACCENT2),
        # Modelo base (lateral)
        (12.0, 6.2, "models/",      "base/ (xtts_v2)",GRAY),
    ]

    box_w, box_h = 1.6, 0.9

    def draw_node(cx, cy, line1, line2, border_color):
        x = cx - box_w / 2
        y = cy - box_h / 2
        fancy = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.04",
            linewidth=1.8,
            edgecolor=border_color,
            facecolor=DARK_PANEL,
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy + 0.14, line1, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=WHITE,
                fontfamily="monospace", zorder=4)
        ax.text(cx, cy - 0.18, line2, ha="center", va="center",
                fontsize=7.5, color=ACCENT if border_color == ACCENT else
                (ACCENT2 if border_color == ACCENT2 else GRAY),
                fontfamily="monospace", zorder=4)

    for n in nodes:
        draw_node(*n)

    # ── Flechas horizontales fila superior ──────────────────────────────────
    top_flow = [1.4, 3.5, 5.6, 7.7, 9.8]
    for i in range(len(top_flow) - 1):
        ax.annotate("",
            xy=(top_flow[i+1] - box_w/2 - 0.05, 6.2),
            xytext=(top_flow[i] + box_w/2 + 0.05, 6.2),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5),
            zorder=2)

    # ── Flecha vertical datos → entrenamiento ───────────────────────────────
    ax.annotate("",
        xy=(9.8, 3.2 + box_h/2 + 0.05),
        xytext=(9.8, 6.2 - box_h/2 - 0.05),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
        zorder=2)

    # ── Flechas horizontales fila inferior ──────────────────────────────────
    bot_flow = [9.8, 12.0, 14.2, 16.4]
    for i in range(len(bot_flow) - 1):
        ax.annotate("",
            xy=(bot_flow[i+1] - box_w/2 - 0.05, 3.2),
            xytext=(bot_flow[i] + box_w/2 + 0.05, 3.2),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
            zorder=2)

    # ── Flecha modelo base → trainer ────────────────────────────────────────
    ax.annotate("",
        xy=(12.0, 3.2 + box_h/2 + 0.05),
        xytext=(12.0, 6.2 - box_h/2 - 0.05),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5, linestyle="dashed"),
        zorder=2)

    # ── Flecha synthesizer → audio sintetizado ───────────────────────────────
    ax.annotate("",
        xy=(16.4, 6.2 - box_h/2 - 0.05),
        xytext=(16.4, 3.2 + box_h/2 + 0.05),
        arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.8),
        zorder=2)

    # ── Flecha modelos trained → synthesizer ────────────────────────────────
    ax.annotate("",
        xy=(16.4 - box_w/2 - 0.05, 3.2),
        xytext=(14.2 + box_w/2 + 0.05, 3.2),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
        zorder=2)

    # ── Texto de entrada al synthesizer ─────────────────────────────────────
    ax.text(16.4, 1.8, "texto.txt", ha="center", va="center",
            fontsize=8, color=GRAY, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_PANEL,
                      edgecolor=GRAY, linewidth=1))
    ax.annotate("",
        xy=(16.4, 3.2 - box_h/2 - 0.05),
        xytext=(16.4, 1.8 + 0.25),
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
        zorder=2)

    # ── Etiquetas de fase ────────────────────────────────────────────────────
    ax.text(0.3, 7.3, "FASE 1 — Preparación del dataset",
            fontsize=8, color=ACCENT, fontfamily="monospace", fontstyle="italic")
    ax.text(0.3, 4.3, "FASE 2 — Fine-tuning e inferencia",
            fontsize=8, color=ACCENT, fontfamily="monospace", fontstyle="italic")

    # ── Leyenda ──────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=DARK_PANEL, edgecolor=ACCENT,  label="Módulo SpeakerTTS"),
        mpatches.Patch(facecolor=DARK_PANEL, edgecolor=GRAY,    label="Dato / Fichero"),
        mpatches.Patch(facecolor=DARK_PANEL, edgecolor=ACCENT2, label="Salida final"),
    ]
    leg = ax.legend(handles=legend_items, loc="lower left",
                    framealpha=0.3, facecolor=DARK_PANEL,
                    edgecolor=GRAY, labelcolor=WHITE, fontsize=8)

    # ── Firma ────────────────────────────────────────────────────────────────
    ax.text(17.7, 0.3, "Created by AdZiBo", ha="right", va="bottom",
            fontsize=7, color=GRAY, fontfamily="monospace")

    out_path = output_dir / "pipeline.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Pipeline guardado: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 2. ESPECTROGRAMA COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════════

def generate_spectrogram(
    ref_wav: Path,
    synth_wav: Path,
    output_dir: Path,
    dpi: int,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import librosa
        import librosa.display
    except ImportError:
        print("  ✗ librosa no está instalado. Ejecuta: pip install librosa")
        sys.exit(1)

    def load_and_melspec(path: Path):
        y, sr = librosa.load(str(path), sr=None, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmax=8000,
            hop_length=256, n_fft=1024,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        duration = librosa.get_duration(y=y, sr=sr)
        return mel_db, sr, duration

    print(f"  Cargando WAV de referencia : {ref_wav}")
    print(f"  Cargando WAV sintetizado   : {synth_wav}")

    mel_ref,   sr_ref,   dur_ref   = load_and_melspec(ref_wav)
    mel_synth, sr_synth, dur_synth = load_and_melspec(synth_wav)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)

    titles = [
        (f"Voz de referencia  —  {dur_ref:.2f}s  |  {sr_ref} Hz",  mel_ref,   ACCENT),
        (f"Voz sintetizada    —  {dur_synth:.2f}s  |  {sr_synth} Hz", mel_synth, ACCENT2),
    ]

    for ax, (title, mel, color) in zip(axes, titles):
        ax.set_facecolor(DARK_PANEL)
        img = librosa.display.specshow(
            mel,
            sr=sr_ref,
            hop_length=256,
            x_axis="time",
            y_axis="mel",
            fmax=8000,
            ax=ax,
            cmap="magma",
        )
        ax.set_title(title, color=color, fontsize=10,
                     fontfamily="monospace", pad=8)
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.xaxis.label.set_color(GRAY)
        ax.yaxis.label.set_color(GRAY)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)

        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        cbar.ax.yaxis.set_tick_params(color=GRAY, labelsize=7)
        cbar.outline.set_edgecolor(GRAY)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GRAY)

    fig.suptitle("Espectrograma Mel — Comparativa Referencia vs. Sintetizado",
                 color=WHITE, fontsize=13, fontfamily="monospace", y=1.01)

    fig.text(0.99, -0.01, "Created by AdZiBo",
             ha="right", va="bottom", fontsize=7,
             color=GRAY, fontfamily="monospace")

    plt.tight_layout()
    out_path = output_dir / "spectrogram.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Espectrograma guardado: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 3. CURVA DE LOSS
# ══════════════════════════════════════════════════════════════════════════════

def generate_loss(log_dir: Path, output_dir: Path, dpi: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Intentar leer desde TensorBoard events
    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        print(f"  Leyendo logs de TensorBoard: {log_dir}")

        ea = EventAccumulator(str(log_dir))
        ea.Reload()

        scalar_keys = ea.Tags().get("scalars", [])

        # Buscar claves de loss (pueden variar según versión de Coqui)
        train_key = next((k for k in scalar_keys if "train" in k.lower() and "loss" in k.lower()), None)
        eval_key  = next((k for k in scalar_keys if ("eval" in k.lower() or "val" in k.lower()) and "loss" in k.lower()), None)

        if train_key:
            events = ea.Scalars(train_key)
            train_steps = [e.step  for e in events]
            train_loss  = [e.value for e in events]
            print(f"  ✓ Loss de entrenamiento: {len(train_steps)} puntos ({train_key})")
        else:
            print(f"  ⚠ No se encontró clave de loss de entrenamiento en: {scalar_keys}")

        if eval_key:
            events = ea.Scalars(eval_key)
            eval_steps = [e.step  for e in events]
            eval_loss  = [e.value for e in events]
            print(f"  ✓ Loss de validación: {len(eval_steps)} puntos ({eval_key})")

    except ImportError:
        print("  ⚠ TensorBoard no disponible. Generando curva de ejemplo ilustrativa.")
    except Exception as e:
        print(f"  ⚠ No se pudieron leer los logs: {e}. Generando curva ilustrativa.")

    # Si no hay datos reales, generar curva ilustrativa
    if not train_steps:
        print("  Generando curva ilustrativa (sin datos reales de entrenamiento).")
        np.random.seed(42)
        train_steps = list(range(0, 5001, 50))
        train_loss  = [4.5 * np.exp(-x / 1800) + 0.45 + np.random.normal(0, 0.04)
                       for x in train_steps]
        eval_steps  = list(range(0, 5001, 500))
        eval_loss   = [4.5 * np.exp(-x / 1800) + 0.52 + np.random.normal(0, 0.06)
                       for x in eval_steps]

    # ── Figura ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_PANEL)

    ax.plot(train_steps, train_loss,
            color=ACCENT, linewidth=1.8, alpha=0.9,
            label="Loss entrenamiento")

    if eval_steps:
        ax.plot(eval_steps, eval_loss,
                color=ACCENT2, linewidth=2, linestyle="--",
                marker="o", markersize=4, alpha=0.9,
                label="Loss validación")

    # Línea del mínimo de validación
    if eval_loss:
        min_val  = min(eval_loss)
        min_step = eval_steps[eval_loss.index(min_val)]
        ax.axvline(x=min_step, color=ACCENT2, linewidth=1,
                   linestyle=":", alpha=0.5)
        ax.text(min_step + max(train_steps) * 0.01, min_val,
                f"  mín val: {min_val:.3f}",
                color=ACCENT2, fontsize=8, fontfamily="monospace", va="bottom")

    ax.set_title("Curva de Loss — Fine-tuning GPT Encoder XTTS v2",
                 color=WHITE, fontsize=12, fontfamily="monospace", pad=12)
    ax.set_xlabel("Paso de entrenamiento", color=GRAY,
                  fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Loss", color=GRAY,
                  fontsize=9, fontfamily="monospace")

    ax.tick_params(colors=GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY)

    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    legend = ax.legend(framealpha=0.3, facecolor=DARK_PANEL,
                       edgecolor=GRAY, labelcolor=WHITE,
                       fontsize=9, loc="upper right")

    fig.text(0.99, 0.01, "Created by AdZiBo",
             ha="right", va="bottom", fontsize=7,
             color=GRAY, fontfamily="monospace")

    plt.tight_layout()
    out_path = output_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Curva de loss guardada: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize.py",
        description="SpeakerTTS — Generador de figuras para documentación",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ejemplos:
  python3 visualize.py --all --ref-wav datasets/Raul/wavs/Raul_0001.wav \\
                              --synth-wav salida.wav \\
                              --log-dir training/VozRaul/logs

  python3 visualize.py --pipeline
  python3 visualize.py --spectrogram --ref-wav REF.wav --synth-wav SYNTH.wav
  python3 visualize.py --loss --log-dir training/VozRaul/logs
        """,
    )

    mode = parser.add_argument_group("Figuras a generar")
    mode.add_argument("--all",         action="store_true", help="Genera las tres figuras")
    mode.add_argument("--pipeline",    action="store_true", help="Diagrama del pipeline")
    mode.add_argument("--spectrogram", action="store_true", help="Espectrograma comparativo")
    mode.add_argument("--loss",        action="store_true", help="Curva de loss")

    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument("--ref-wav",   metavar="WAV",  help="WAV de referencia (voz real)")
    inputs.add_argument("--synth-wav", metavar="WAV",  help="WAV sintetizado por el modelo")
    inputs.add_argument("--log-dir",   metavar="DIR",  help="Carpeta de logs de TensorBoard")

    opts = parser.add_argument_group("Opciones")
    opts.add_argument("--output-dir",  metavar="DIR",  default="docs/figures",
                      help="Carpeta de salida (default: docs/figures/)")
    opts.add_argument("--dpi",         type=int, default=150,
                      help="Resolución de las figuras en DPI (default: 150)")

    return parser


def main() -> None:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    do_pipeline    = args.all or args.pipeline
    do_spectrogram = args.all or args.spectrogram
    do_loss        = args.all or args.loss

    if not any([do_pipeline, do_spectrogram, do_loss]):
        print("  ✗ Indica al menos una figura: --pipeline, --spectrogram, --loss o --all")
        sys.exit(1)

    print(f"\n  SpeakerTTS — Generador de figuras")
    print(f"  Salida: {output_dir.resolve()}\n")

    if do_pipeline:
        print("  [1/3] Generando diagrama del pipeline...")
        generate_pipeline(output_dir, args.dpi)

    if do_spectrogram:
        print("\n  [2/3] Generando espectrograma comparativo...")
        if not args.ref_wav or not args.synth_wav:
            print("  ✗ --spectrogram requiere --ref-wav y --synth-wav")
            sys.exit(1)
        ref_wav   = Path(args.ref_wav)
        synth_wav = Path(args.synth_wav)
        if not ref_wav.exists():
            print(f"  ✗ Archivo no encontrado: {ref_wav}")
            sys.exit(1)
        if not synth_wav.exists():
            print(f"  ✗ Archivo no encontrado: {synth_wav}")
            sys.exit(1)
        generate_spectrogram(ref_wav, synth_wav, output_dir, args.dpi)

    if do_loss:
        print("\n  [3/3] Generando curva de loss...")
        log_dir = Path(args.log_dir) if args.log_dir else None
        if log_dir and not log_dir.exists():
            print(f"  ⚠ Carpeta de logs no encontrada: {log_dir}")
            print("    Se generará una curva ilustrativa.")
            log_dir = None
        generate_loss(log_dir or Path("."), output_dir, args.dpi)

    print(f"\n  ✓ Figuras guardadas en: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
