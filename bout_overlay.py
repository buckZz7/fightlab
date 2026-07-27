"""Add HP bars, round timer, and fighter names as overlay on bout videos.
Post-process the rendered MP4 with text overlays using PIL."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def add_overlay(frame_img, hp_red, hp_blue, round_num, round_time, name_red, name_blue):
    """Draw HP bars + timer on a frame."""
    img = frame_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # HP bar dimensions
    bar_w = int(w * 0.35)
    bar_h = 8
    bar_y = 20
    bar_margin = 20

    # Red (left) HP bar
    red_w = int(bar_w * max(0, hp_red) / 100)
    draw.rectangle([bar_margin, bar_y, bar_margin + bar_w, bar_y + bar_h],
                   fill=(40, 40, 40), outline=(80, 80, 80))
    draw.rectangle([bar_margin, bar_y, bar_margin + red_w, bar_y + bar_h],
                   fill=(220, 60, 60))

    # Blue (right) HP bar
    blue_w = int(bar_w * max(0, hp_blue) / 100)
    blue_x = w - bar_margin - bar_w
    draw.rectangle([blue_x, bar_y, blue_x + bar_w, bar_y + bar_h],
                   fill=(40, 40, 40), outline=(80, 80, 80))
    draw.rectangle([blue_x + bar_w - blue_w, bar_y, blue_x + bar_w, bar_y + bar_h],
                   fill=(60, 120, 240))

    # Fighter names
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((bar_margin, bar_y - 18), name_red[:20], fill=(220, 60, 60), font=font)
    draw.text((blue_x, bar_y - 18), name_blue[:20], fill=(60, 120, 240), font=font)

    # Round + timer (center)
    timer_text = f"R{round_num} {round_time:.1f}s"
    bbox = draw.textbbox((0, 0), timer_text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((w // 2 - tw // 2, 10), timer_text, fill=(255, 255, 255), font=font)

    return img


def overlay_video(input_mp4, output_mp4, name_red="Red", name_blue="Blue",
                  hp_red=100, hp_blue=100, rounds=3, round_seconds=30.0):
    """Add HP bar overlay to a rendered bout video."""
    import imageio_ffmpeg
    import subprocess
    import tempfile

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    frames_dir = tempfile.mkdtemp()

    # Extract frames
    subprocess.run([ff, "-i", input_mp4, "-vf", "fps=30",
                    f"{frames_dir}/f%05d.png"], capture_output=True)

    # Get frame count
    frames = sorted(os.listdir(frames_dir))
    total_frames = len(frames)
    if not frames:
        print("No frames extracted")
        return

    # Add overlay to each frame
    fps = 30
    for i, fname in enumerate(frames):
        img = Image.open(os.path.join(frames_dir, fname))
        t = i / fps
        round_num = min(int(t / round_seconds) + 1, rounds)
        round_time = t % round_seconds
        overlayed = add_overlay(img, hp_red, hp_blue, round_num, round_time,
                                 name_red, name_blue)
        overlayed.save(os.path.join(frames_dir, fname))

    # Re-encode
    subprocess.run([ff, "-y", "-framerate", "30", "-i", f"{frames_dir}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    output_mp4], capture_output=True)

    # Cleanup
    import shutil
    shutil.rmtree(frames_dir)
    print(f"[overlay] saved {output_mp4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--red", default="Red")
    ap.add_argument("--blue", default="Blue")
    ap.add_argument("--hp-red", type=float, default=100)
    ap.add_argument("--hp-blue", type=float, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--round-seconds", type=float, default=30.0)
    a = ap.parse_args()
    overlay_video(a.input, a.output, a.red, a.blue, a.hp_red, a.hp_blue,
                  a.rounds, a.round_seconds)
