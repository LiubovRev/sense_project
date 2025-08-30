#!/usr/bin/env python3

import subprocess
from pathlib import Path
import sys
import math
import traceback

# ================================================= 
# ==== Custom settings ===
# ================================================= 
chunk_duration_sec = 120   # duration of each chunk in seconds
overlap_sec = 0           # overlap between chunks
nb_objects = "2"
yolo_model = "yolo11m.pt"
model_size = "base_plus"
scale_factor = 0.8        # 80% of original resolution

# === Optional video trimming settings ===
start_trim_sec = 155    #or None
end_trim_sec = None     #or None


base_path = "/home/liubov/Bureau/new"
raw_video = f"{base_path}/output_crf24.mp4"
chunk_dir = Path(f"{base_path}/Chunks")
chunk_dir.mkdir(parents=True, exist_ok=True)

log_file_path = Path(f"{base_path}/processing_log.log")
log_file_path.write_text("=== Processing Log ===\n")  # Clear or initialize log

def log_to_file(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")



# ================================================= 
# ============ Utility Functions ==================
# ================================================= 

def run_cmd(cmd, step_desc):
    print(f"\n>>> {step_desc}")
    print("Running command:", " ".join(map(str, cmd)))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Command succeeded.")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {step_desc} failed with exit code {e.returncode}")
        print("Command:", " ".join(e.cmd))
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise RuntimeError(f"{step_desc} failed with exit code {e.returncode}:\n{e.stderr}") from e


def get_video_duration(video_path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Failed to get video duration.")
        print(result.stderr)
        sys.exit(1)

    return float(result.stdout.strip())

# ================================================= 
# ============ Step 0: Trim input video (optional) ============
# ================================================= 

trimmed_video = raw_video  # default to raw video if no trimming

if start_trim_sec is not None and end_trim_sec is not None:
    trimmed_video = f"{base_path}/trimmed_input.mp4"
    trim_duration = end_trim_sec - start_trim_sec

    ffmpeg_trim_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_trim_sec),    # Seek start time
        "-i", raw_video,               # Input is original raw video
        "-t", str(trim_duration),      # Duration to trim
        "-vf", f"scale=iw*{scale_factor}:ih*{scale_factor}",  # Scale here to save re-scaling later
        "-c:v", "libx264",             # Re-encode video for accurate trimming
        "-c:a", "aac",                 # Re-encode audio
        trimmed_video
    ]

    run_cmd(ffmpeg_trim_cmd, "Trimming input video")

# ================================================= 
# ============ Step 1: Split trimmed video into chunks ============
# ================================================= 

video_duration = get_video_duration(trimmed_video)
stride = chunk_duration_sec - overlap_sec
num_chunks = math.ceil((video_duration - overlap_sec) / stride)

chunk_paths = []

for i in range(num_chunks):
    start = i * stride
    end = min(start + chunk_duration_sec, video_duration)
    actual_duration = end - start

    time_range = f"{int(start):05d}-{int(end):05d}s"
    output_chunk = chunk_dir / f"chunk_{i:02d}_{time_range}.mp4"
    chunk_paths.append(output_chunk)

    ffmpeg_chunk_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),              # Seek start time of chunk (before input for fast seek)
        "-i", trimmed_video,
        "-t", str(actual_duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_chunk)
    ]

    run_cmd(ffmpeg_chunk_cmd, f"Splitting chunk {i+1}/{num_chunks} (start: {start:.2f}s, duration: {actual_duration:.2f}s)")


# ================================================= 
# ============ Step 1.2: Process Each Chunk ========
# ================================================= 

for idx, processed_video in enumerate(chunk_paths):
    print(f"\n================== Processing chunk {idx+1}/{num_chunks}: {processed_video.name} ==================")

    # Extract time range from filename
    time_label = processed_video.stem.split("_")[-1]

    # Output directories
    mask_dir = f"{base_path}/MaskDir/chunk_{idx:02d}_{time_label}"
    poses_dir = f"{base_path}/PosesDir/chunk_{idx:02d}_{time_label}"
    faces_dir = f"{base_path}/FacesDir/chunk_{idx:02d}_{time_label}"
    vis_tracking = f"{base_path}/Visualizations/Visualization_tracking_chunk_{idx:02d}_{time_label}.mp4"
    vis_pose = f"{base_path}/Visualizations/Visualization_pose_chunk_{idx:02d}_{time_label}.mp4"
    vis_face = f"{base_path}/Visualizations/Visualization_face_chunk_{idx:02d}_{time_label}.mp4"

    # Create output directories
    for d in [mask_dir, poses_dir, faces_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # === Step 2: Tracking inference ===
    tracking_cmd = [
        "psifx", "video", "tracking", "samurai", "inference",
        "--video", str(processed_video),
        "--mask_dir", mask_dir,
        "--model_size", model_size,
        "--yolo_model", yolo_model,
        "--object_class", "0",
        "--max_objects", nb_objects,
        "--step", "30",
        "--device", "cuda",
        "--overwrite"
    ]

    try:
        run_cmd(tracking_cmd, f"Chunk {idx+1}/{num_chunks}: Tracking inference")
    except Exception as e:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        last_relevant_line = tb_lines[-1].strip()
        msg = (
        f"============[SKIPPED] Chunk {idx+1:02d} ({processed_video.name}): "
        f"Tracking failed ({last_relevant_line})\n===================="
    )
        print(msg)
        log_to_file(msg)
        continue


    # === Step 3: Visualize tracking ===
    run_cmd([
        "psifx", "video", "tracking", "visualization",
        "--video", str(processed_video),
        "--masks", mask_dir,
        "--visualization", vis_tracking,
        "--no-blackout",
        "--labels",
        "--color",
        "--overwrite"
    ], f"Chunk {idx+1}/{num_chunks}: Visualizing tracking")

    # === Step 4: Pose inference ===
    run_cmd([
        "psifx", "video", "pose", "mediapipe", "multi-inference",
        "--video", str(processed_video),
        "--masks", mask_dir,
        "--poses_dir", poses_dir,
        "--device", "cuda",
        "--overwrite"
    ], f"Chunk {idx+1}/{num_chunks}: Pose inference")

    # === Step 5: Visualize pose ===
    run_cmd([
        "psifx", "video", "pose", "mediapipe", "visualization",
        "--video", str(processed_video),
        "--poses", poses_dir,
        "--visualization", vis_pose,
        "--confidence_threshold", "0.0",
        "--overwrite"
    ], f"Chunk {idx+1}/{num_chunks}: Visualizing pose")

    # === Step 6: Face feature extraction ===
    run_cmd([
        "psifx", "video", "face", "openface", "multi-inference",
        "--video", str(processed_video),
        "--masks", mask_dir,
        "--features_dir", faces_dir,
        "--device", "cuda",
        "--overwrite"
    ], f"Chunk {idx+1}/{num_chunks}: Face inference")

    # === Step 7: Visualize face features ===
    run_cmd([
        "psifx", "video", "face", "openface", "visualization",
        "--video", str(processed_video),
        "--features", faces_dir,
        "--visualization", vis_face,
        "--depth", "3.0",
        "--f_x", "1600.0", "--f_y", "1600.0",
        "--c_x", "960.0", "--c_y", "540.0",
        "--overwrite"
    ], f"Chunk {idx+1}/{num_chunks}: Visualizing face")

    # === Log success ===
    msg = f"===========[OK] Chunk {idx+1:02d} ({processed_video.name}): Processed successfully\n===================="
    log_to_file(msg)

print("\n All chunks processed. See processing_log.txt for details.")
