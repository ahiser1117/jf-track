#!/usr/bin/env python3
"""Batch-processing helper for jf-track videos."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import click


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--pattern', default='*.mp4', help='Glob pattern for videos')
@click.option('--output-root', default='./batch_results', help='Directory to store results')
@click.option('--cli-args', default='', help='Additional arguments to pass to src.cli')
def batch_process(directory: str, pattern: str, output_root: str, cli_args: str) -> None:
    """Process every video inside DIRECTORY using the main CLI."""

    directory_path = Path(directory)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    videos = sorted(directory_path.glob(pattern))
    if not videos:
        click.echo(f"No videos found in {directory_path} matching pattern '{pattern}'.")
        return

    click.echo(f"Processing {len(videos)} videos from {directory_path}...")

    for video in videos:
        video_output = output_root_path / video.stem
        video_output.mkdir(parents=True, exist_ok=True)
        cmd = f"python -m src.cli \"{video}\" --output \"{video_output}\" {cli_args}"
        click.echo(f"\n➡️  Running: {cmd}")
        try:
            subprocess.run(shlex.split(cmd), check=True)
        except subprocess.CalledProcessError as exc:
            click.echo(f"❌ Failed to process {video}: {exc}")


if __name__ == '__main__':
    batch_process()
