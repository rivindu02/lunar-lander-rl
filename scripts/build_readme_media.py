#!/usr/bin/env python3
"""Build compact README GIFs from the original recorded simulator episodes.

Requires ffmpeg/ffprobe (including drawtext); downloads nothing, trains nothing.
The original videos and their result records are never changed.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets/readme"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(run, index):
    folder = ROOT / "videos" / run
    video = folder / f"lander-episode-{index}.mp4"
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = next(item["sha256"] for item in manifest["files"]
                    if item["path"] == str(video.relative_to(ROOT)))
    if digest(video) != expected:
        raise ValueError(f"Source video checksum mismatch: {video}")
    with (folder / "episodes.csv").open(newline="") as handle:
        row = list(csv.DictReader(handle))[index]
    return video, row


def panel(input_id, title, color, score):
    return (f"[{input_id}:v]fps=12,scale=420:280:flags=lanczos,"
            "tpad=stop_mode=clone:stop_duration=2,trim=duration=6.94,"
            "setpts=PTS-STARTPTS,pad=420:350:0:44:color=0x0b1220,"
            f"drawtext=font='DejaVu Sans':text='{title}':x=14:y=13:"
            f"fontsize=17:fontcolor={color},"
            f"drawtext=font='DejaVu Sans':text='Episode return  {score:.2f}':"
            "x=14:y=328:fontsize=13:fontcolor=0xcbd5e1")


def gif(inputs, graph, output):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
    for path in inputs:
        command += ["-i", str(path)]
    command += ["-filter_complex", graph + ";[frames]split[a][b];"
                "[a]palettegen=max_colors=128:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle[out]",
                "-map", "[out]", "-loop", "0", str(output)]
    subprocess.run(command, check=True)


def main():
    for program in ("ffmpeg", "ffprobe"):
        if not shutil.which(program):
            raise RuntimeError(f"Install {program} before building README animations")
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, base_row = episode("final_baseline_seed42", 0)
    shaped, shaped_row = episode("final_shaped_seed42", 0)
    timeout, timeout_row = episode("final_baseline_seed42", 2)
    comparison = OUT / "landing-comparison.gif"
    gif([baseline, shaped],
        panel(0, "BASELINE PPO", "0x60a5fa", float(base_row["return"])) + "[left];" +
        panel(1, "TILT-SHAPED PPO", "0x34d399", float(shaped_row["return"])) + "[right];" +
        "[left][right]hstack=inputs=2[frames]", comparison)
    failure = OUT / "baseline-timeout.gif"
    gif([timeout],
        "[0:v]setpts=0.5*(PTS-STARTPTS),fps=10,scale=420:280:flags=lanczos,"
        "pad=420:350:0:44:color=0x0b1220,"
        "drawtext=font='DejaVu Sans':text='BASELINE TIMEOUT   /   2x speed':"
        "x=14:y=13:fontsize=16:fontcolor=0xfbbf24,"
        f"drawtext=font='DejaVu Sans':text='Return {float(timeout_row['return']):.2f}   /   1000-step time limit':"
        "x=14:y=328:fontsize=12:fontcolor=0xcbd5e1[frames]", failure)
    preview = OUT / "landing-preview.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "3", "-i", str(comparison), "-frames:v", "1", str(preview)], check=True)
    sources = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                "episode_seed": int(row["episode_seed"]), "return": float(row["return"])}
               for path, row in [(baseline, base_row), (shaped, shaped_row), (timeout, timeout_row)]]
    data = {"training_seed": 42, "sources": sources,
            "comparison": "12 fps; full demo seed 30000 at original speed; shorter episode holds final frame",
            "timeout": "10 fps; full demo seed 30002 at 2x playback speed",
            "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                         "bytes": path.stat().st_size} for path in (comparison, failure, preview)]}
    (OUT / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    for path in (comparison, failure):
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
