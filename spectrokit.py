import statistics
import typer
from pathlib import Path
import random
import json
import os

# enable caching for librosa
os.environ["LIBROSA_CACHE_DIR"] = "tmp/librosa_cache"

import librosa
import features
import visualize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


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

def process_file(audio_file: Path, functions: list[str], labels: list[str], duration: float, image_output: Path) -> dict:
    """
    Process a single audio file and return analysis results.
    """
    waveform, sr = librosa.load(audio_file, sr=None, duration=duration)
    analysis_result = {}
    
    for funcname in functions:
        try:
            func = getattr(features, funcname)
            value = func(waveform, sr)
            analysis_result[funcname] = float(value)
        except Exception as e:
            raise RuntimeError(f"Error running function '{funcname}' on file '{audio_file}': {e}") from e
    
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
    functions: str = typer.Option(..., help="List of analysis functions to run as a single string, separated by spaces. Supported: " + ", ".join(features.__all__)),
    sample_size: int = typer.Option(None, help="Random sample size (optional)."),
    labels: str = typer.Option(None, help="List of labels matching input files as a single string, separated by spaces."),
    duration: float = typer.Option(None, help="Max duration in seconds to analyze."),
    output: Path = typer.Option("./", help="Where to write results."),
    image_output: Path = typer.Option(
        None, 
        help="Where to store generated spectrograms for each audio file. If not provided, spectrograms will not be saved."
    ),
    workers: int = typer.Option(None, help="Number of parallel workers to use for processing. Defaults to number of CPU cores."),
    seed: int = typer.Option(None, help="Random seed for reproducibility. If not provided, a random seed will be used.")
):
    # discover files
    if input.is_file():
        audio_files = [input]
    else:
        audio_files = find_audio_files(input)
    
    if not audio_files:
        typer.echo("No audio files found.")
        raise typer.Exit(1)
    
    if seed is not None:
        random.seed(seed)
        typer.echo(f"Using random seed: {seed}")

    # apply sampling if needed
    if sample_size:
        audio_files = random.sample(audio_files, min(sample_size, len(audio_files)))

    # split functions string into a list
    functions = functions.split()

    # split labels string into a list if provided
    if labels:
        labels = labels.split()

    # discover available functions from __all__
    available_funcs = discover_analysis_functions()

    # validate requested functions
    for funcname in functions:
        if funcname not in available_funcs:
            typer.echo(f"Function '{funcname}' is not listed in features.__all__.")
            raise typer.Exit(1)

    results = []
    pbar = tqdm(total=len(audio_files), desc="Processing")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_file, audio_file, functions, labels, duration, image_output)
            for audio_file in audio_files
        ]
        for f in as_completed(futures):
            try:
                result = f.result()
                results.append(result)
                pbar.write(f"Processed {result['file']}")
            except Exception as e:
                pbar.write(f"Error processing file: {e}")
            pbar.update(1)
    pbar.close()
    typer.echo(f"Processed {len(results)} files.")

    with open(os.path.join(output, f"{'_'.join(labels)}-results.json"), "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(f"Analysis complete. Results saved to '{output}'")

    # calculate and print summary statistics
    if results:
        summary = {func: [] for func in functions}
        for result in results:
            for func, value in result['analysis'].items():
                summary[func].append(value)

        typer.echo("\nSummary Statistics:")
        for func, values in summary.items():
            if values:
                mean_value = sum(values) / len(values)
                typer.echo(f"{func}: Mean = {mean_value:.4f}, Standard Deviation = {statistics.stdev(values):.4f}, Count = {len(values)}")
            else:
                typer.echo(f"{func}: No data available")
    # clear cache
    librosa.cache.clear()
    

if __name__ == "__main__":
    app()
