import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 1. JOINT MAPPING
# ==============================
JOINT_NAMES = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip",
    9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip",
    13: "LKnee", 14: "LAnkle", 15: "REye", 16: "LEye",
    17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe",
    21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
    25: "Background"
}

# ==============================
# 2. LOAD MERGED JSON
# ==============================
def load_merged_skeletons(json_path, person_id):
    """Load merged skeleton JSON for one person into a DataFrame."""
    with open(json_path, "r") as f:
        data = json.load(f)

    all_frames, all_skeletons = [], []

    for frame_idx, frame_data in enumerate(data):
        skeleton = np.array(frame_data["pose_keypoints_2d"], dtype=np.float32)
        all_frames.append(frame_idx)
        all_skeletons.append(skeleton)

    df = pd.DataFrame({
        "frame": all_frames,
        "id_name": person_id,
        "skeleton": all_skeletons
    })
    return df

# ==============================
# 3. EXTRACT JOINT POSITIONS
# ==============================
def extract_joint_positions(skeleton_dataset, person_id, joint_idx=0, confidence_threshold=0.5):
    joint_name = JOINT_NAMES.get(joint_idx, f"Joint_{joint_idx}")
    person_data = skeleton_dataset[skeleton_dataset["id_name"] == person_id].copy()

    frames = person_data["frame"].values
    positions = {}

    for frame, skeleton in zip(frames, person_data["skeleton"].values):
        if skeleton is None or len(skeleton) < (joint_idx*3+3):
            continue
        x, y, conf = skeleton[joint_idx*3: joint_idx*3+3]
        if conf >= confidence_threshold:
            positions[frame] = (x, y)

    return positions, joint_name

# ==============================
# 4. MOVEMENT VARIANCE
# ==============================
def compute_variance(positions):
    if not positions:
        return None
    coords = np.array(list(positions.values()))
    var_x, var_y = np.var(coords, axis=0)
    return var_x, var_y

# ==============================
# 5. PLOT TRAJECTORY (Optional)
# ==============================
def plot_joint_trajectory(positions, joint_name, person_id):
    coords = np.array(list(positions.values()))
    plt.plot(coords[:,0], coords[:,1], marker="o", markersize=1, linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.title(f"{person_id} - {joint_name} Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# ==============================
# 6. MAIN EXECUTION
# ==============================
if __name__ == "__main__":

    # Paths to your JSONs
    base_dir = "/home/liubov/Desktop/BNF/work/8-5-2024_#18_INDIVIDUAL_[12]/PosesDir"
    patient_files = {
        "Patient1": os.path.join(base_dir, "new_jsons_merged_id_1.json"),
        "Patient2": os.path.join(base_dir, "new_jsons_merged_id_2.json"),
        "Patient3": os.path.join(base_dir, "new_jsons_merged_id_3.json"),
    }

    # Load all patients
    datasets = []
    for pid, path in patient_files.items():
        print(f"Loading {pid} from {path}")
        df = load_merged_skeletons(path, pid)
        datasets.append(df)

    skeleton_dataset = pd.concat(datasets, ignore_index=True)

    # Choose which joints to analyze
    joints_to_check = list(JOINT_NAMES.keys())
    results = []

    for pid in patient_files.keys():
        for j_idx in joints_to_check:
            positions, joint_name = extract_joint_positions(skeleton_dataset, pid, joint_idx=j_idx)
            var = compute_variance(positions)
            if var:
                results.append({
                    "Patient": pid,
                    "Joint": joint_name,
                    "X": var[0],
                    "Y": var[1]
                })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    out_csv = "joint_movement_variance.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    print(results_df)

# ==============================
# 7. SAVE FRAME-LEVEL POSITIONS
# ==============================
all_positions = []

for pid in patient_files.keys():
    for j_idx in joints_to_check:
        positions, joint_name = extract_joint_positions(
            skeleton_dataset, pid, joint_idx=j_idx
        )
        if positions:
            for frame, (x, y) in positions.items():
                all_positions.append({
                    "Patient": pid,
                    "Joint": joint_name,
                    "Frame": frame,
                    "X": x,
                    "Y": y
                })

positions_df = pd.DataFrame(all_positions)
positions_csv = "joint_positions.csv"
positions_df.to_csv(positions_csv, index=False)
print(f"Frame-level positions saved to {positions_csv}")

