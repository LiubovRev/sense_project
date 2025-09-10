import numpy as np
import os
import cv2
import mediapipe as mp
import json
import argparse
from PIL import Image
from collections import defaultdict, Counter
import utils
import torch
from torchvision import transforms
import pandas as pd
from torchreid.utils import FeatureExtractor
from sklearn.cluster import KMeans



def extract_patch_info(filename):
    parts = filename.split('_')
    patch_id = int(parts[1])
    frame_number = int(parts[3].split('.')[0])
    return patch_id, frame_number


def save_pose_landmarks(pose_landmarks, output_path):
    with open(output_path, 'w') as f:
        json.dump(pose_landmarks, f)


def convert_landmarks_to_pixel_coordinates(image, landmarks):
    image_height, image_width, _ = image.shape
    pixel_landmarks = []
    for landmark in landmarks.landmark:
        pixel_landmarks.append({
            'x': int(landmark.x * image_width),
            'y': int(landmark.y * image_height),
            'visibility:': (landmark.visibility)
        })
    return pixel_landmarks


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_patches_by_skeletons(skeleton_dir, patch_dir):
    # checks visibility of skeleton joins and removes corresponding skeleton and patches if too low
    for patch_file in os.listdir(patch_dir):
        if patch_file.endswith('_masked.png'):
            base_name = patch_file.replace('_masked.png', '')
            json_file = f'{base_name}_keypoints.json'
            json_path = os.path.join(skeleton_dir, json_file)

            if not os.path.exists(json_path):
                patch_path = os.path.join(patch_dir, patch_file)
                os.remove(patch_path)
                continue

            with open(json_path, 'r') as f:
                keypoints_2d = json.load(f)

            neck_visibility = keypoints_2d[1].get('visibility:', 0)
            rshoulder_visibility = keypoints_2d[2].get('visibility:', 0)
            lshoulder_visibility = keypoints_2d[5].get('visibility:', 0)
            rear_visibility = keypoints_2d[15].get('visibility:', 0)
            lear_visibility = keypoints_2d[16].get('visibility:', 0)

            if neck_visibility + rshoulder_visibility + lshoulder_visibility + rear_visibility + lear_visibility < 0.1:
                os.remove(json_path)
                patch_path = os.path.join(patch_dir, patch_file)
                if os.path.exists(patch_path):
                    os.remove(patch_path)


def get_torchreid_image_embeddings(image_path, transform, extractor):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(extractor.device)
    with torch.no_grad():
        embedding = extractor(image)
    return embedding.cpu().numpy().flatten()


def embed_patches(patches_dir, transform, extractor):
    embeddings = []
    patch_paths = []

    for patch_file in os.listdir(patches_dir):
        patch_path = os.path.join(patches_dir, patch_file)
        if os.path.isfile(patch_path) and patch_file.endswith('_masked.png'):
            embedding = get_torchreid_image_embeddings(patch_path, transform, extractor)
            embeddings.append(embedding)
            patch_paths.append(patch_file)

    return np.array(embeddings), patch_paths


def cluster_embeddings(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto', max_iter=300, tol=0.0001, init='k-means++').fit(embeddings)
    return kmeans.labels_


def ensure_unique_ids_within_frames(patch_paths, labels):
    frame_to_labels = defaultdict(list)
    for patch_file, label in zip(patch_paths, labels):
        patch_id, frame_number = extract_patch_info(os.path.basename(patch_file))
        frame_to_labels[frame_number].append((patch_file, label))

    updated_patch_paths = []
    updated_labels = []

    for frame_number, items in frame_to_labels.items():
        labels_in_frame = [label for _, label in items]
        label_counts = Counter(labels_in_frame)
        duplicates = [label for label, count in label_counts.items() if count > 1]

        if duplicates:
            filtered_items = [item for item in items if label_counts[item[1]] == 1]
        else:
            filtered_items = items

        for patch_file, label in filtered_items:
            updated_patch_paths.append(patch_file)
            updated_labels.append(label)

    return updated_patch_paths, updated_labels


def create_id_mapping(patch_paths, labels):
    mapping = []
    for patch_file, label in zip(patch_paths, labels):
        original_id, frame_number = extract_patch_info(os.path.basename(patch_file))
        mapping.append([frame_number, original_id, label])
    return pd.DataFrame(mapping, columns=['frame_number', 'original_id', 'updated_id'])


def initialize_mediapipe():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    return pose, selfie_segmentation


def process_image_patch(patch_path, pose, selfie_segmentation, dark_pixel_threshold=0.1, enable_skeleton_detection=True):
    # Check if the patch is already masked (we can estimate skeletons more than  once, but only want to mask once)
    if '_masked' in patch_path:
        image_rgb = cv2.imread(patch_path)
        if image_rgb is None:
            print(f"Error reading image {patch_path}")
            return None, None

        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb) if enable_skeleton_detection else None
        return image_rgb, pose_results

    # Otherwise, read the image using OpenCV
    image = cv2.imread(patch_path)
    if image is None:
        print(f"Error reading image {patch_path}")
        return None, None

    # Convert the image from BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image for segmentation
    mask_results = selfie_segmentation.process(image_rgb)

    if mask_results.segmentation_mask is not None:
        segmentation_mask = mask_results.segmentation_mask
        condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.5
        masked_image = np.where(condition, image_rgb, 0)

        non_zero_pixels = np.count_nonzero(masked_image)
        total_pixels = masked_image.shape[0] * masked_image.shape[1] * masked_image.shape[2]
        non_zero_ratio = non_zero_pixels / total_pixels

        if non_zero_ratio < dark_pixel_threshold:
            return None, None

        masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

        pose_results = pose.process(image_rgb) if enable_skeleton_detection else None
        return masked_image_bgr, pose_results

    return None, None


def save_masked_patch_and_keypoints(patch_path, masked_image, pose_results, skeleton_dir, enable_skeleton_detection):
    patch_id, frame_number = extract_patch_info(os.path.basename(patch_path))
    if enable_skeleton_detection and pose_results.pose_landmarks:
        pixel_landmarks = convert_landmarks_to_pixel_coordinates(masked_image, pose_results.pose_landmarks)
        keypoints_filename = f'patch_{patch_id}_frame_{frame_number}_keypoints.json'
        keypoints_path = os.path.join(skeleton_dir, keypoints_filename)
        save_pose_landmarks(pixel_landmarks, keypoints_path)

    if '_masked' not in patch_path:
        masked_patch_filename = patch_path.replace('.png', '_masked.png')
        cv2.imwrite(masked_patch_filename, masked_image)
        os.remove(patch_path)


def process_patches(video_dir, config, enable_skeleton_detection):
    patches_dir = os.path.join(video_dir, config['PATCHES_DIR'], config['VIDEO_DIR'], config['CAMERA_NAME'].split('.')[0])
    skeleton_dir = os.path.join(video_dir, config['COORD2d_DIR'], config['VIDEO_DIR'], config['CAMERA_NAME'].split('.')[0])

    create_directory(skeleton_dir)

    pose, selfie_segmentation = initialize_mediapipe()
    all_patch_filenames = os.listdir(patches_dir)

    for patch_filename in all_patch_filenames:
        patch_path = os.path.join(patches_dir, patch_filename)
        masked_image, pose_results = process_image_patch(patch_path, pose, selfie_segmentation, 0.2, enable_skeleton_detection)
        if masked_image is not None:
            save_masked_patch_and_keypoints(patch_path, masked_image, pose_results, skeleton_dir, enable_skeleton_detection)

    if enable_skeleton_detection:
        filter_patches_by_skeletons(skeleton_dir, patches_dir)
    selfie_segmentation.close()
    pose.close()


def get_mapping(video_dir, config):
    patches_dir = os.path.join(video_dir, config['PATCHES_DIR'], config['VIDEO_DIR'], config['CAMERA_NAME'].split('.')[0])
    mapping_dir = os.path.join(video_dir)
    mapping_path = os.path.join(video_dir, mapping_dir, f'first_id_mapping_{config["CAMERA_NAME"].split(".")[0]}.csv')

    create_directory(mapping_dir)

    torchreid_extractor = FeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path='models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    torchreid_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    embeddings, patch_paths = embed_patches(patches_dir, torchreid_transform, torchreid_extractor)
    labels = cluster_embeddings(embeddings, config['MAX_N'])
    updated_patch_paths, updated_labels = ensure_unique_ids_within_frames(patch_paths, labels)
    id_mapping_df = create_id_mapping(updated_patch_paths, updated_labels)
    id_mapping_df.to_csv(mapping_path, index=False)
    print(f"ID mapping saved to {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patches for skeletons and embeddings.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--enable_skeleton_detection', action='store_true', help='Enable skeleton detection.')
    args = parser.parse_args()

    print('Getting the configuration file...')
    config = utils.load_config(config_file=args.config)

    video_dir = os.path.join(config.get('ROOT_DIR'), config.get('VIDEO_DIR'))

    print('Getting skeletons and masking patches...')
    process_patches(video_dir, config, args.enable_skeleton_detection)

    print('Running reidentification algorithm...')
    get_mapping(video_dir, config)
