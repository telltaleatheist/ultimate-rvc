"""
Module which defines the command-line interface for generating
audio using RVC.
"""

from __future__ import annotations

from typing import Annotated

import time

# NOTE typer actually uses Path from pathlib at runtime
# even though it appears it is only a type annotation
from pathlib import Path  # noqa: TC003

import typer
from rich import print as rprint
from rich.panel import Panel

from ultimate_rvc.cli.common import (
    complete_audio_ext,
    complete_embedder_model,
    complete_f0_method,
    format_duration,
)
# BookForge fork: the song-cover and speech subcommands are omitted. They eagerly
# pull web-UI / TTS / stem-separation deps (anyio, edge-tts, audio-separator,
# yt-dlp) that this inference-only env does not ship. Only `convert-voice` and
# `wavify` (defined below) are kept.
from ultimate_rvc.cli.typing_extra import PanelName
from ultimate_rvc.core.generate.common import convert as _convert
from ultimate_rvc.core.generate.common import wavify as _wavify
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method, RVCContentType

app = typer.Typer(
    name="generate",
    no_args_is_help=True,
    help="Generate audio using RVC",
    rich_markup_mode="markdown",
)


# BookForge fork: song_cover_app / speech_app intentionally not registered.


@app.command(no_args_is_help=True)
def wavify(
    audio_track: Annotated[
        Path,
        typer.Argument(
            help="The path to the audio track to convert.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    directory: Annotated[
        Path,
        typer.Argument(
            help=(
                "The path to the directory where the converted audio track will be"
                " saved."
            ),
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    prefix: Annotated[
        str,
        typer.Argument(
            help="The prefix to use for the name of the converted audio track.",
        ),
    ],
    accepted_format: Annotated[
        list[AudioExt] | None,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_audio_ext,
            help=(
                "An audio format to accept for conversion. This option can be provided"
                " multiple times to accept multiple formats. If not provided, the"
                " default accepted formats are mp3, ogg, flac, m4a and aac."
            ),
        ),
    ] = None,
) -> None:
    """
    Convert an audio track to wav format if its current format is an
    accepted format.
    """
    start_time = time.perf_counter()

    rprint()

    wav_path = _wavify(
        audio_track=audio_track,
        directory=directory,
        prefix=prefix,
        accepted_formats=set(accepted_format) if accepted_format else None,
    )
    if wav_path == audio_track:
        rprint(
            "[+] Audio track was not converted to WAV format. Presumably, "
            "its format is not in the given list of accepted formats.",
        )
    else:
        rprint("[+] Audio track succesfully converted to WAV format!")
        rprint()
        rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
        rprint(Panel(f"[green]{wav_path}", title="WAV Audio Track Path"))


@app.command(no_args_is_help=True)
def convert_voice(
    voice_track: Annotated[
        Path,
        typer.Argument(
            help="The path to the voice track to convert.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    directory: Annotated[
        Path,
        typer.Argument(
            help=(
                "The path to the directory where the converted voice track will be"
                " will be saved."
            ),
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    model_name: Annotated[
        str,
        typer.Argument(
            help="The name of the model to use for voice conversion.",
        ),
    ],
    n_octaves: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.MAIN_OPTIONS,
            help=(
                "The number of octaves to pitch-shift the converted voice by. Use"
                " 1 for male-to-female and -1 for vice-versa."
            ),
        ),
    ] = 0,
    n_semitones: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.MAIN_OPTIONS,
            help=(
                "The number of semi-tones to pitch-shift the converted"
                " voice by. Altering this slightly reduces sound quality."
            ),
        ),
    ] = 0,
    f0_method: Annotated[
        list[F0Method] | None,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_f0_method,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "The method to use for pitch extraction. This"
                " option can be provided multiple times to use multiple pitch"
                " extraction methods in combination. If not provided, will default to"
                " the rmvpe method, which is generally recommended."
            ),
        ),
    ] = None,
    index_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "The rate of influence of the index file. Increase to bias the"
                " conversion towards the accent of the voice model. Decrease to"
                " potentially reduce artifacts."
            ),
        ),
    ] = 0.3,
    rms_mix_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "Blending rate for the volume envelope of the converted voice. Controls"
                " how much to mimic the loudness of the input voice (0) or a fixed"
                " loudness (1)."
            ),
        ),
    ] = 1.0,
    protect_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=0.5,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "A coefficient which controls the extent to which consonants and"
                " breathing sounds are protected from artifacts. A higher value"
                " offers more protection but may worsen the indexing"
                " effect."
            ),
        ),
    ] = 0.33,
    hop_length: Annotated[
        int,
        typer.Option(
            min=1,
            max=512,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "Controls how often the CREPE-based pitch extraction method checks for"
                " pitch changes measured in milliseconds. Lower values lead to longer"
                " conversion times and a higher risk of voice cracks, but better pitch"
                " accuracy."
            ),
        ),
    ] = 128,
    split_voice: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.VOICE_ENRICHMENT_OPTIONS,
            help=(
                "Whether to split the voice track into smaller segments"
                " before converting it. This can improve output quality for"
                " longer voice tracks."
            ),
        ),
    ] = False,
    autotune_voice: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.VOICE_ENRICHMENT_OPTIONS,
            help="Whether to apply autotune to the converted voice.",
        ),
    ] = False,
    autotune_strength: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOICE_ENRICHMENT_OPTIONS,
            help=(
                "The intensity of the autotune effect to apply to the converted voice."
                " Higher values result in stronger snapping to the chromatic grid."
            ),
        ),
    ] = 1.0,
    clean_voice: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.VOICE_ENRICHMENT_OPTIONS,
            help=(
                "Whether to clean the converted voice using noise reduction algorithms"
            ),
        ),
    ] = False,
    clean_strength: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOICE_ENRICHMENT_OPTIONS,
            help=(
                "The intensity of the cleaning to apply to the converted voice. Higher"
                " values result in stronger cleaning, but may lead to a more compressed"
                " sound."
            ),
        ),
    ] = 0.7,
    embedder_model: Annotated[
        EmbedderModel,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_embedder_model,
            rich_help_panel=PanelName.SPEAKER_EMBEDDINGS_OPTIONS,
            help="The model to use for generating speaker embeddings.",
        ),
    ] = EmbedderModel.CONTENTVEC,
    custom_embedder_model: Annotated[
        str | None,
        typer.Option(
            rich_help_panel=PanelName.SPEAKER_EMBEDDINGS_OPTIONS,
            help=(
                "The name of a custom embedder model to use for generating speaker"
                " embeddings. Only applicable if `embedder-model` is set to `custom`."
            ),
        ),
    ] = None,
    sid: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.SPEAKER_EMBEDDINGS_OPTIONS,
            help="The id of the speaker to use for multi-speaker RVC models.",
        ),
    ] = 0,
) -> None:
    """Convert a voice track using RVC."""
    start_time = time.perf_counter()

    rprint()

    converted_voice_path = _convert(
        audio_track=voice_track,
        directory=directory,
        model_name=model_name,
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_methods=f0_method,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        split_audio=split_voice,
        autotune_audio=autotune_voice,
        autotune_strength=autotune_strength,
        clean_audio=clean_voice,
        clean_strength=clean_strength,
        embedder_model=embedder_model,
        custom_embedder_model=custom_embedder_model,
        sid=sid,
        content_type=RVCContentType.VOICE,
    )
    rprint("[+] Voice track succesfully converted!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{converted_voice_path}", title="Converted Voice Path"))


@app.command(no_args_is_help=True)
def convert_dir(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory of sentence audio files to convert.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory to write converted files into (created if missing).",
            resolve_path=True,
        ),
    ],
    model_name: Annotated[
        str,
        typer.Argument(help="The name of the RVC model to use for voice conversion."),
    ],
    n_semitones: Annotated[
        int,
        typer.Option(help="Semitones to pitch-shift the converted voice by."),
    ] = 0,
    f0_method: Annotated[
        list[F0Method] | None,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_f0_method,
            help="Pitch extraction method(s). Defaults to rmvpe.",
        ),
    ] = None,
    index_rate: Annotated[
        float,
        typer.Option(min=0, max=1, help="Rate of influence of the index file."),
    ] = 0.3,
    rms_mix_rate: Annotated[
        float,
        typer.Option(min=0, max=1, help="Volume-envelope blend rate."),
    ] = 1.0,
    protect_rate: Annotated[
        float,
        typer.Option(min=0, max=0.5, help="Consonant/breath protection coefficient."),
    ] = 0.33,
    hop_length: Annotated[
        int,
        typer.Option(min=1, max=512, help="CREPE pitch-check interval (ms)."),
    ] = 128,
    embedder_model: Annotated[
        EmbedderModel,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_embedder_model,
            help="Speaker-embedding model.",
        ),
    ] = EmbedderModel.CONTENTVEC,
    custom_embedder_model: Annotated[
        str | None,
        typer.Option(help="Custom embedder model name (when embedder-model=custom)."),
    ] = None,
    sid: Annotated[
        int,
        typer.Option(help="Speaker id for multi-speaker RVC models."),
    ] = 0,
    input_glob: Annotated[
        str,
        typer.Option(help="Glob selecting input files within input-dir."),
    ] = "*.flac",
    output_ext: Annotated[
        str,
        typer.Option(help="Output file extension / format (e.g. flac, wav)."),
    ] = "flac",
    overwrite: Annotated[
        bool,
        typer.Option(help="Reconvert files even if already present in output-dir."),
    ] = False,
) -> None:
    """
    Batch-convert every audio file in a directory with ONE model load (warm model).

    BookForge fork: converts a whole directory of TTS sentence files in a single
    process. The voice converter is a cached singleton and the model is only
    reloaded when the model path changes, so the RVC model + embedder are loaded
    once and reused for every file — making per-sentence conversion practical.
    Output files keep the input stem (e.g. ``0.flac`` -> ``0.flac``) so e2a's
    assembler can consume them via ``--sentences_dir``. Emits ``[RVC] done/total``
    lines (flushed) for external progress tracking. Files already present in
    output-dir are skipped unless ``--overwrite``.
    """
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import soundfile as sf  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)

    def sort_key(p: Path) -> tuple[int, str]:
        return (int(p.stem), "") if p.stem.isdigit() else (1 << 62, p.stem)

    files = sorted(input_dir.glob(input_glob), key=sort_key)
    total = len(files)
    rprint(f"[RVC-BATCH] {total} file(s), model={model_name}, index_rate={index_rate}")
    print(f"[RVC] 0/{total}", flush=True)

    start_time = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="urvc_batch_"))
    done = 0
    converted = 0
    try:
        for f in files:
            out_path = output_dir / f"{f.stem}.{output_ext}"
            if out_path.exists() and not overwrite:
                done += 1
                print(f"[RVC] {done}/{total}", flush=True)
                continue
            result_path = _convert(
                audio_track=f,
                directory=tmp,
                model_name=model_name,
                n_octaves=0,
                n_semitones=n_semitones,
                f0_methods=f0_method,
                index_rate=index_rate,
                rms_mix_rate=rms_mix_rate,
                protect_rate=protect_rate,
                hop_length=hop_length,
                split_audio=False,
                autotune_audio=False,
                autotune_strength=1.0,
                clean_audio=False,
                clean_strength=0.7,
                embedder_model=embedder_model,
                custom_embedder_model=custom_embedder_model,
                sid=sid,
                content_type=RVCContentType.VOICE,
            )
            # Re-container into the expected stem/format for the assembler, then
            # drop the intermediate so the temp dir doesn't grow unbounded.
            data, sr = sf.read(str(result_path))
            sf.write(str(out_path), data, sr)
            try:
                os.remove(result_path)
            except OSError:
                pass
            converted += 1
            done += 1
            print(f"[RVC] {done}/{total}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rprint(
        f"[+] Batch complete: {converted} converted, {total - converted} skipped, "
        f"in {format_duration(time.perf_counter() - start_time)}",
    )
    rprint(Panel(f"[green]{output_dir}", title="Converted Sentences Directory"))
