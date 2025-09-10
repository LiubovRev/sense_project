import os
import pandas as pd
import cv2
import argparse
import json
import utils
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def calculate_centroid(bbox):
    """ Calculate the centroid of a bounding box. """
    xmin, ymin, xmax, ymax = bbox[:4]
    x_centroid = (xmin + xmax) / 2
    y_centroid = (ymin + ymax) / 2
    return x_centroid, y_centroid


def get_bounding_boxes(frame_number, box_dir):
    """ Load bounding boxes for a specific frame number. """
    bbox_file = os.path.join(box_dir, f'boxes_{frame_number}.txt')
    bboxes = []
    if os.path.exists(bbox_file):
        with open(bbox_file, 'r') as f:
            for line in f:
                bbox = list(map(float, line.strip().split()))  # Assumes each line is xmin ymin xmax ymax confidence
                bboxes.append(bbox)
    return bboxes


def extract_trajectories(mapping, box_dir, start_frame, end_frame):
    """ Extract trajectories from bounding boxes and the ID mapping. """
    trajectories = {}

    for frame in range(start_frame, end_frame + 1):
        bboxes = get_bounding_boxes(frame, box_dir)

        if not bboxes:
            continue

        if frame not in mapping['frame_number'].values:
            print(f"Frame {frame} not found in the mapping file, skipping.")
            continue

        frame_mapping = mapping[mapping['frame_number'] == frame]

        for initial_id, bbox in enumerate(bboxes):
            updated_id_row = frame_mapping[frame_mapping['original_id'] == initial_id]

            if not updated_id_row.empty:
                updated_id = updated_id_row['updated_id'].values[0]
                centroid = calculate_centroid(bbox)

                if updated_id not in trajectories:
                    trajectories[updated_id] = {'x': [], 'y': [], 'frames': []}

                # Append centroid and frame number to the trajectory
                trajectories[updated_id]['x'].append(centroid[0])
                trajectories[updated_id]['y'].append(centroid[1])
                trajectories[updated_id]['frames'].append(frame)

    return trajectories



def smooth_trajectory(data, window_size=3):
    """ Apply moving average smoothing to the trajectory data. """
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

def smooth_trajectories(trajectories, window_size=3):
    """ Smooth the x and y coordinates of all trajectories. """
    smoothed_trajectories = {}

    for updated_id, trajectory in trajectories.items():
        smoothed_trajectories[updated_id] = {
            'x': smooth_trajectory(trajectory['x'], window_size),
            'y': smooth_trajectory(trajectory['y'], window_size),
            'frames': trajectory['frames'][window_size-1:]  # Adjust frame list for the valid part of smoothing
        }

    return smoothed_trajectories


def get_video_resolution(video_path):
    """ Get video resolution (width, height) using OpenCV. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return width, height


def export_trajectories_to_csv(trajectories, output_file):
    """ Export the trajectories to a CSV file. Each row contains the updated_id, x, y, and frame. """
    rows = []

    # Loop over each updated ID and its associated trajectory
    for updated_id, trajectory in trajectories.items():
        for i in range(len(trajectory['x'])):
            row = {
                'updated_id': updated_id,
                'x': trajectory['x'][i],
                'y': trajectory['y'][i],
                'frame': trajectory['frames'][i]
            }
            rows.append(row)

    # Convert the rows to a DataFrame
    df = pd.DataFrame(rows)

    # Export to CSV
    df.to_csv(output_file, index=False)
    print(f"Trajectories successfully exported to {output_file}")


def plot_3d_trajectories(trajectories, VIDEO_DIR, frame_scale_factor=5):
    """ Plot 3D trajectories with x and y as pixel coordinates and z as time in minutes.
        The z-axis will be scaled to be longer than x and y by a frame_scale_factor.
    """
    fps = 30
    fig = plt.figure(figsize=(8, 6))  # Adjust figure size for better fit
    ax = fig.add_subplot(111, projection='3d')

    # Customize colors for different trajectories
    colors = plt.get_cmap('tab10', len(trajectories)) # Use a 10-color tabular colormap

    # Scale the z-axis down but make it longer than x and y
    z_scale_factor = 2  # Adjust this factor to make the z-axis visually longer

    min_frame = min([min(trajectory['frames']) for trajectory in trajectories.values()])

    for idx, (updated_id, trajectory) in enumerate(trajectories.items()):
        x = trajectory['x']
        y = trajectory['y']
        z = [(frame - min_frame) / z_scale_factor for frame in trajectory['frames']]

        # Customize trace style: thicker lines, color cycling, and transparency
        ax.plot(z, y, x, label=f'ID {updated_id}', color=colors(idx), linewidth=2, alpha=0.8)

    # Adjust aspect ratio and zoom in on the plot
    ax.set_box_aspect([frame_scale_factor * 2, 1, 1])  # Double the scaling for Z (now X)

    # Calculate time in minutes for Z-axis (now represented as X)
    max_frame = max([max(trajectory['frames']) for trajectory in trajectories.values()]) - min_frame

    # Define regular minute ticks (e.g., 5, 10, 15, etc.)
    max_minutes = (max_frame // (fps * 60)) + 1  # Convert to minutes and round up
    minute_ticks = np.arange(0, max_minutes * (fps * 60), 2 * fps * 60)  # 5-minute intervals
    z_ticks = [t / z_scale_factor for t in minute_ticks]  # Scale to match plot
    z_labels = [str(int(t // (fps * 60))) for t in minute_ticks]  # Convert ticks to minutes

    # Set ticks and labels for the new X-axis (originally Z-axis)
    ax.set_xticks(z_ticks)
    ax.set_xticklabels(z_labels)

    # Improve the appearance of the axes
    ax.grid(True, linestyle='--', alpha=0.6)  # Light dashed gridlines
    ax.set_xlabel('Time (minutes)', fontsize=12, labelpad=30)
    # Remove yticks and zticks
    ax.set_yticks([])
    ax.set_zticks([])

    # Set axis limits for a more centered view (optional, adjust as needed)
    ax.set_xlim(min(z_ticks), max(z_ticks))

    # Set a less vertical, more dynamic viewing angle
    ax.view_init(elev=20, azim=240)

    # Use tight_layout to remove excess white space
    plt.tight_layout()

    # Optionally, manually adjust subplot margins
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save the figure
    plt.savefig(f'plots/{VIDEO_DIR}.png', dpi=200)


def save_frame_from_video(video_path, frame_number, output_path):
    """ Save a specific frame from the video to a file. """
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap.read()

    if not success:
        raise ValueError(f"Unable to read frame {frame_number} from {video_path}")

    # Save the frame as an image file
    cv2.imwrite(output_path, frame)

    # Release the video capture object
    cap.release()

    print(f"Frame {frame_number} saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect people and get bounding boxes.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = utils.load_config(config_file=args.config)

    video_dir = os.path.join(config.get('ROOT_DIR'), config.get('VIDEO_DIR'))
    VIDEO_DIR = config.get('VIDEO_DIR')
    CAMERA_NAME = config.get('CAMERA_NAME')
    FRAME_DIR = config.get('FRAME_DIR')
    BOX_DIR = config.get('BOUNDINGBOX_DIR')
    PATCHES_DIR = config.get('PATCHES_DIR')
    START_FRAME = config.get('START_FRAME', 0)
    END_FRAME = config.get('END_FRAME', 1000)
    ID_TO_NAME_MAPPING = config.get('ID_TO_NAME_MAPPING', {})

    camera_name = CAMERA_NAME.split('.')[0]
    mapping_path = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, f'first_id_mapping_{camera_name}.csv')
    box_dir = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, BOX_DIR, VIDEO_DIR, camera_name)
    frame_dir = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, FRAME_DIR, VIDEO_DIR, camera_name)
    video_dir = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR)
    video_path = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, 'camera_a_with_ids_openpose.mp4')

    mapping = pd.read_csv(mapping_path)
    start_offset = 1000
    end_offset = 10000

    video_width, video_height = get_video_resolution(video_path)
    trajectories = extract_trajectories(mapping, box_dir, START_FRAME + start_offset, END_FRAME - end_offset)

    # Save the first frame
    first_frame_output_path = f'plots/{VIDEO_DIR}_first.png'
    save_frame_from_video(video_path, mapping['frame_number'].min() - START_FRAME + start_offset, first_frame_output_path)

    # Save the last frame
    last_frame_output_path = f'plots/{VIDEO_DIR}_last.png'
    save_frame_from_video(video_path, mapping['frame_number'].max() - START_FRAME - end_offset, last_frame_output_path)

    # Trajectory processing
    window_size = 20
    smoothed_trajectories = smooth_trajectories(trajectories, window_size)

    # Plot smoothed trajectories
    plot_3d_trajectories(smoothed_trajectories, VIDEO_DIR, frame_scale_factor=3.5)