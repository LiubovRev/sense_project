import os
import pandas as pd
import cv2
import argparse
import json
import utils

def draw_bounding_boxes(frame, bboxes, ids, id_to_name_mapping):
    for bbox, id in zip(bboxes, ids):
        x1, y1, x2, y2 = map(int, bbox[:4])
        name = id_to_name_mapping.get(str(id), str(id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue box with thickness 1
        cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return frame

# OpenPose connections
OPENPOSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)
])


def draw_skeletons(frame, skeletons, bboxes):
    for skeleton, bbox in zip(skeletons, bboxes):
        if skeleton is not None:
            x_offset, y_offset = int(bbox[0]), int(bbox[1])
            joint_positions = []
            for i in range(0, len(skeleton), 3):
                x, y = int(skeleton[i] + x_offset), int(skeleton[i+1] + y_offset)
                joint_positions.append((x, y))
                if int(skeleton[i]) != 0 and int(skeleton[i+1]) != 0:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Green circle for skeleton point

            for connection in OPENPOSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_pos = joint_positions[start_idx]
                end_pos = joint_positions[end_idx]
                # Avoid drawing connections that link to the origin (0,0)
                if start_pos != (x_offset, y_offset) and end_pos != (x_offset, y_offset):
                    if (start_idx < len(joint_positions) and end_idx < len(joint_positions) and
                            start_pos is not None and end_pos is not None):
                        cv2.line(frame, start_pos, end_pos, (255, 0, 0), 1)  # Blue line for connections
    return frame

def read_skeletons(skeleton_dir, frame_number):
    skeletons = []
    skeleton_files = [f for f in os.listdir(skeleton_dir) if f'frame_{frame_number}_masked_keypoints' in f]

    sorted_skeleton_files = sorted(skeleton_files, key=lambda x: int(x.split('_')[1]))

    for skeleton_file in sorted_skeleton_files:
        skeleton_path = os.path.join(skeleton_dir, skeleton_file)
        with open(skeleton_path, 'r') as f:
            data = json.load(f)
            if 'people' in data and len(data['people']) > 0:
                skeleton = data['people'][0]['pose_keypoints_2d']
                skeletons.append(skeleton)
            else:
                skeletons.append(None)

    return skeletons

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
    SKELETON_DIR = config.get('COORD2d_DIR', 'skeletons')
    ID_TO_NAME_MAPPING = config.get('ID_TO_NAME_MAPPING', {})

    camera_name = CAMERA_NAME.split('.')[0]
    mapping_path = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, f'first_id_mapping_{camera_name}.csv')
    box_dir = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, BOX_DIR, VIDEO_DIR, camera_name)
    frame_dir = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, FRAME_DIR, VIDEO_DIR, camera_name)
    video_dir = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR)
    skeleton_dir = os.path.join(config.get('ROOT_DIR'), VIDEO_DIR, SKELETON_DIR, VIDEO_DIR, camera_name)

    mapping = pd.read_csv(mapping_path)

    frames = {}
    print('Collecting frames and creating video...')
    for _, row in mapping.iterrows():
        frame_number = row['frame_number']
        original_id = row['original_id']
        updated_id = row['updated_id']

        box_file = os.path.join(box_dir, f'boxes_{frame_number}.txt')
        if frame_number not in frames:
            frames[frame_number] = []

        if os.path.exists(box_file):
            with open(box_file, 'r') as f:
                boxes = [list(map(float, line.strip().split())) for line in f]

                skeletons = read_skeletons(skeleton_dir, frame_number)

                skeletons = skeletons + [None] * (len(boxes) - len(skeletons))

                # Ensure the skeletons correspond to the correct original_id
                skeleton_for_id = skeletons[original_id] if original_id < len(skeletons) else None

                frames[frame_number].append((boxes[original_id], updated_id, skeleton_for_id))


    sample_frame_path = os.path.join(frame_dir, f'frame_{START_FRAME}.png')
    print(sample_frame_path)
    sample_frame = cv2.imread(sample_frame_path)
    frame_height, frame_width, _ = sample_frame.shape

    # Define the codec and create a VideoWriter object
    output_video_path = os.path.join(video_dir, f'{camera_name}_with_ids_openpose.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    print('Processing each frame starting from frame:', START_FRAME)
    for frame_idx in range(START_FRAME, END_FRAME):
        frame_path = os.path.join(frame_dir, f'frame_{frame_idx}.png')

        if not os.path.exists(frame_path):
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue  # Skip if the frame does not exist

        if frame_idx in frames:
            bboxes = [item[0] for item in frames[frame_idx]]
            ids = [item[1] for item in frames[frame_idx]]
            frame = draw_bounding_boxes(frame, bboxes, ids, ID_TO_NAME_MAPPING)
            skeletons = [item[2] for item in frames[frame_idx]]
            if skeletons is not None:
                frame = draw_skeletons(frame, skeletons, bboxes)

        cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    # Release the video writer
    out.release()

    print(f"Video saved to {output_video_path}")

# POSE_BODY_25_BODY_PARTS {
# //     {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
# //     {8,  "MidHip"},
# //     {9,  "RHip"},
# //     {10, "RKnee"},
# //     {11, "RAnkle"},
# //     {12, "LHip"},
# //     {13, "LKnee"},
# //     {14, "LAnkle"},
# //     {15, "REye"},
# //     {16, "LEye"},
# //     {17, "REar"},
# //     {18, "LEar"},
# //     {19, "LBigToe"},
# //     {20, "LSmallToe"},
# //     {21, "LHeel"},
# //     {22, "RBigToe"},
# //     {23, "RSmallToe"},
# //     {24, "RHeel"},
# //     {25, "Background"}