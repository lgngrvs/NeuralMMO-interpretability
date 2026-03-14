#!/usr/bin/env python
"""Render a NeuralMMO replay file as a GIF.

Usage:
    # From an existing replay file:
    python render_replay.py replay_file.replay.lzma -o output.gif

    # Generate a new replay and render:
    python render_replay.py --policy takeru_100M -o output.gif

    # Options:
    #   --fps 10          frames per second (default 10)
    #   --skip N          render every Nth tick (default 1)
    #   --size 600        output image size in pixels (default 600)
"""

import argparse
import glob
import io
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nmmo.render.replay_helper import FileReplayHelper

# Tile material ID -> RGB color
TILE_COLORS = {
    0: (0.0, 0.0, 0.0),       # void / unpassable -> black
    1: (0.25, 0.45, 0.85),    # water -> blue
    2: (0.45, 0.75, 0.35),    # grass -> green
    3: (0.20, 0.50, 0.20),    # foliage/forest -> dark green
    4: (0.55, 0.55, 0.55),    # stone -> gray
    5: (0.85, 0.20, 0.15),    # lava/death fog -> red
}

# Fixed palette for player colors (reused cyclically)
PLAYER_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def generate_replay_file(policy_name, policies_dir="policies"):
    """Run train.py -m replay to generate a replay, return the replay file path.

    The policy pool expects a directory of .pt files. We create a temp dir with
    a symlink to the selected policy, matching extract_activations.py's approach.
    """
    import shutil
    import tempfile

    src = os.path.join(os.path.abspath(policies_dir), f"{policy_name}.pt")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Policy not found: {src}")

    tmp_dir = tempfile.mkdtemp(prefix="nmmo_replay_")
    try:
        os.symlink(src, os.path.join(tmp_dir, f"{policy_name}.pt"))

        # Find existing replay files before generation
        before = set(glob.glob(os.path.join(tmp_dir, "*.replay.lzma")))

        cmd = [sys.executable, "train.py", "-m", "replay", "-p", tmp_dir,
               "--train.device", "cpu"]
        print(f"Running: {' '.join(cmd)}")
        env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        subprocess.run(cmd, check=True, env=env)

        # Find the new replay file
        after = set(glob.glob(os.path.join(tmp_dir, "*.replay.lzma")))
        new_files = after - before
        if not new_files:
            raise RuntimeError(f"No new .replay.lzma found in {tmp_dir}")

        # Copy it out of the temp dir before cleanup
        replay_src = max(new_files, key=os.path.getmtime)
        replay_dst = os.path.join("replays", os.path.basename(replay_src))
        os.makedirs("replays", exist_ok=True)
        shutil.copy2(replay_src, replay_dst)
        return replay_dst
    finally:
        shutil.rmtree(tmp_dir)


def load_replay(replay_path):
    """Load a replay file using nmmo's FileReplayHelper."""
    print(f"Loading replay from {replay_path} ...")
    replay = FileReplayHelper.load(replay_path)
    print(f"  Map size: {len(replay.map)}x{len(replay.map[0])}")
    print(f"  Ticks: {len(replay.packets)}")
    return replay


def build_tile_image(tile_map):
    """Convert the 2D tile material ID array to an RGB numpy array."""
    h, w = len(tile_map), len(tile_map[0])
    img = np.zeros((h, w, 3), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            tid = tile_map[r][c]
            img[r, c] = TILE_COLORS.get(tid, (0.0, 0.0, 0.0))
    return img


def get_player_color(agent_id, color_map):
    """Return a consistent color for a given agent_id."""
    if agent_id not in color_map:
        color_map[agent_id] = PLAYER_PALETTE[len(color_map) % len(PLAYER_PALETTE)]
    return color_map[agent_id]


def render_frame(tile_img, packet, tick, color_map, fig_size_px):
    """Render a single frame and return a PIL Image."""
    dpi = 100
    fig_inches = fig_size_px / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_inches, fig_inches), dpi=dpi)

    map_h, map_w = tile_img.shape[:2]
    ax.imshow(tile_img, extent=[0, map_w, map_h, 0], interpolation="nearest")

    # Draw NPCs as small gray dots
    npcs = packet.get("npc", {})
    for eid, npc_data in npcs.items():
        base = npc_data.get("base", {})
        r, c = base.get("r"), base.get("c")
        if r is not None and c is not None:
            ax.plot(c + 0.5, r + 0.5, "o", color="gray", markersize=2, alpha=0.6)

    # Draw players as colored circles sized by health
    players = packet.get("player", {})
    alive_count = 0
    for eid, pdata in players.items():
        if not pdata.get("alive", False):
            continue
        alive_count += 1
        base = pdata.get("base", {})
        r, c = base.get("r"), base.get("c")
        if r is None or c is None:
            continue

        resources = pdata.get("resources", {})
        health = resources.get("health", {})
        hp_val = health.get("val", 100)
        hp_max = health.get("max", 100)
        hp_frac = hp_val / max(hp_max, 1)

        # Size: 3 to 7 based on health fraction
        ms = 3 + 4 * hp_frac
        color = get_player_color(eid, color_map)
        ax.plot(c + 0.5, r + 0.5, "o", color=color, markersize=ms,
                markeredgecolor="black", markeredgewidth=0.3)

    # HUD text
    ax.text(2, 3, f"Tick {tick}", color="white", fontsize=8,
            fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=2))
    ax.text(2, 8, f"Alive: {alive_count}", color="white", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.set_xlim(0, map_w)
    ax.set_ylim(map_h, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Rasterize to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_gif(replay, output_path, fps=10, skip=1, size=600):
    """Render all frames and assemble into a GIF."""
    tile_img = build_tile_image(replay.map)
    color_map = {}
    frames = []

    ticks = list(range(0, len(replay.packets), skip))
    total = len(ticks)
    print(f"Rendering {total} frames ...")

    for i, tick_idx in enumerate(ticks):
        packet = replay.packets[tick_idx]
        frame = render_frame(tile_img, packet, tick_idx, color_map, size)
        frames.append(frame)
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  [{i + 1}/{total}]")

    if not frames:
        print("No frames to render.")
        return

    duration_ms = int(1000 / fps)
    print(f"Saving GIF to {output_path} ({len(frames)} frames, {duration_ms}ms per frame) ...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Render a NeuralMMO replay as a GIF"
    )
    parser.add_argument(
        "replay_file", nargs="?", default=None,
        help="Path to a .replay.lzma or .replay.json file"
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        help="Policy name (e.g. takeru_100M); generates a replay via train.py -m replay"
    )
    parser.add_argument("-o", "--output", type=str, default="output.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--skip", type=int, default=1, help="Render every Nth tick (default: 1)")
    parser.add_argument("--size", type=int, default=600, help="Output image size in pixels (default: 600)")

    args = parser.parse_args()

    if args.replay_file is None and args.policy is None:
        parser.error("Provide either a replay file or --policy to generate one")

    # Step 1: get the replay file path
    if args.replay_file is not None:
        replay_path = args.replay_file
    else:
        replay_path = generate_replay_file(args.policy)
        print(f"Replay saved to: {replay_path}")

    # Step 2: load and render
    replay = load_replay(replay_path)
    render_gif(replay, args.output, fps=args.fps, skip=args.skip, size=args.size)


if __name__ == "__main__":
    main()
