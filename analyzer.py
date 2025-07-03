import typer
from pathlib import Path
import random
import json
import librosa
import features
import visualize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

app = typer.Typer()

def find_audio_files(directory: Path):
    """Recursively find supported audio files"""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    ]

def discover_analysis_functions():
    """
    Load all functions explicitly listed in features.__all__
    """
    functions = {}
    for name in features.__all__:
        func = getattr(features, name, None)
        if callable(func):
            functions[name] = func
    return functions

def process_file(audio_file: Path, functions: list[str], labels: list[str], duration: float, image_output: Path):
    """
    Process a single audio file and return analysis results.
    """
    typer.echo(f"Processing: {audio_file}")
    waveform, sr = librosa.load(audio_file, sr=None, duration=duration)
    analysis_result = {}
    
    for funcname in functions:
        try:
            func = getattr(features, funcname)
            value = func(waveform, sr)
            analysis_result[funcname] = value
        except Exception as e:
            typer.echo(f"Error running {funcname} on {audio_file.name}: {e}")
            analysis_result[funcname] = None
    
    if image_output is not None:
        visualize.save_spectrogram(
            waveform=waveform,
            sr=sr,
            labels=labels,
            name=audio_file.stem,
            output_dir=image_output
        )
            
    return {
        "file": str(audio_file),
        "labels": labels,
        "analysis": analysis_result
    }


@app.command()
def analyze(
    input: Path = typer.Option(..., help="File or directory containing audio."),
    functions: list[str] = typer.Option(..., help="List of analysis functions to run. Supported: " + ", ".join(features.__all__)),
    sample_size: int = typer.Option(None, help="Random sample size (optional)."),
    labels: list[str] = typer.Option(None, help="List of labels matching input files"),
    duration: float = typer.Option(None, help="Max duration in seconds to analyze."),
    output: Path = typer.Option("results.json", help="Where to write results."),
    image_output: Path = typer.Option(
        None, 
        help="Where to store generated spectrograms for each audio file. If not provided, spectrograms will not be saved."
    ),
    workers: int = typer.Option(None, help="Number of parallel workers to use for processing. Defaults to number of CPU cores.")
):
    # discover files
    if input.is_file():
        audio_files = [input]
    else:
        audio_files = find_audio_files(input)
    
    if not audio_files:
        typer.echo("No audio files found.")
        raise typer.Exit(1)

    # apply sampling if needed
    if sample_size:
        audio_files = random.sample(audio_files, min(sample_size, len(audio_files)))

    # discover available functions from __all__
    available_funcs = discover_analysis_functions()

    # validate requested functions
    for funcname in functions:
        if funcname not in available_funcs:
            typer.echo(f"Function '{funcname}' is not listed in features.__all__.")
            raise typer.Exit(1)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_file, audio_file, functions, labels, duration, image_output)
            for audio_file in audio_files
        ]
        with tqdm(total=len(audio_files), desc="Processing") as pbar:
            for f in as_completed(futures):
                try:
                    result = f.result()
                    results.append(result)
                except Exception as e:
                    typer.echo(f"Error processing file: {e}")
                pbar.update(1)
    
    typer.echo(f"Processed {len(results)} files.")

    with open(os.path.join(output, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"Analysis complete. Results saved to {output}")

if __name__ == "__main__":
    app()
