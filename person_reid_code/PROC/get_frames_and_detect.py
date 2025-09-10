import cv2
import os
import utils
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_frames(video_dir, config):
    video_path = os.path.join(video_dir, config.get('CAMERA_NAME'))
    camera_name = config.get('CAMERA_NAME').split('.')[0]
    frame_dir = os.path.join(os.path.join(video_dir, config.get('FRAME_DIR'), config.get('VIDEO_DIR')), camera_name)

    create_directory(frame_dir)

    recording = cv2.VideoCapture(video_path)
    print(video_path)
    total_frames = int(recording.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = 0

    while current_frame < config.get('END_FRAME', 1000):
        ret, color_image = recording.read()
        if not ret:
            break
        if current_frame >= config.get('START_FRAME', 0):
            frame_filename = os.path.join(frame_dir, f'frame_{current_frame}.png')
            cv2.imwrite(frame_filename, color_image)
        current_frame += 1
        if current_frame % 1000 == 0:
            print(f"Processed {current_frame} frames")

    recording.release()
    cv2.destroyAllWindows()


def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def custom_nms(boxes, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    keep_boxes = []
    while boxes:
        largest_box = boxes.pop(0)
        keep_boxes.append(largest_box)
        boxes = [box for box in boxes if compute_iou(largest_box[:4], box[:4]) < iou_threshold]
    return keep_boxes


def yolov8_detection(model, image, iou_threshold=0.5):
    results = model(image, stream=True, conf=0.3)
    person_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:
                bbox = box.xyxy.tolist()[0]
                confidence = box.conf.tolist()[0]
                person_boxes.append(bbox + [confidence])
    person_boxes = custom_nms(person_boxes, iou_threshold)
    return person_boxes


def detect_people(video_dir, config):
    frame_dir = os.path.join(video_dir, os.path.join(config.get('FRAME_DIR'), config.get('VIDEO_DIR')), config.get('CAMERA_NAME').split('.')[0])
    box_dir = os.path.join(video_dir, os.path.join(config.get('BOUNDINGBOX_DIR'), config.get('VIDEO_DIR')), config.get('CAMERA_NAME').split('.')[0])
    patch_dir = os.path.join(video_dir, os.path.join(config.get('PATCHES_DIR'), config.get('VIDEO_DIR')), config.get('CAMERA_NAME').split('.')[0])

    create_directory(box_dir)
    create_directory(patch_dir)

    all_frame_filenames = os.listdir(frame_dir)

    model = YOLO('yolov8x.pt')

    for frame_name in all_frame_filenames:
        frame_number = int(frame_name.split('_')[1].split('.')[0])
        frame_path = os.path.join(frame_dir, frame_name)
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolov8_boxes = yolov8_detection(model, image)

        if yolov8_boxes:
            yolov8_boxes = sorted(yolov8_boxes, key=lambda x: x[4], reverse=True)[:config.get('MAX_N')]
            box_filename = os.path.join(box_dir, f'boxes_{frame_number}.txt')
            with open(box_filename, 'w') as f:
                for box in yolov8_boxes:
                    x1, y1, x2, y2, confidence = map(str, box)
                    f.write(f'{x1} {y1} {x2} {y2} {confidence}\n')

def extract_patches(video_dir, config):
    frame_dir = os.path.join(video_dir, os.path.join(config.get('FRAME_DIR'), config.get('VIDEO_DIR')), config.get('CAMERA_NAME').split('.')[0])
    box_dir = os.path.join(video_dir, os.path.join(config.get('BOUNDINGBOX_DIR'), config.get('VIDEO_DIR')), config.get('CAMERA_NAME').split('.')[0])
    patch_dir = os.path.join(video_dir, os.path.join(config.get('PATCHES_DIR'), config.get('VIDEO_DIR')), config.get('CAMERA_NAME').split('.')[0])

    create_directory(box_dir)
    create_directory(patch_dir)

    boxes_paths = os.listdir(box_dir)
    for boxes_path in boxes_paths:
        frame_number = int(boxes_path.split('_')[1].split('.')[0])
        box_file_path = os.path.join(box_dir, boxes_path)

        with open(box_file_path, 'r') as f:
            boxes = [list(map(float, line.strip().split())) for line in f]

        frame_path = os.path.join(frame_dir, f'frame_{frame_number}.png')
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: Frame {frame_path} not found.")
            continue

        for idx, bbox in enumerate(boxes):
            x1, y1, x2, y2 = map(int, bbox[:4])
            patch = image[y1:y2, x1:x2]
            patch_filename = os.path.join(patch_dir, f'patch_{idx}_frame_{frame_number}.png')
            cv2.imwrite(patch_filename, patch)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract video frames and detect people.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = utils.load_config(config_file=args.config)

    video_dir = os.path.join(config.get('ROOT_DIR'), config.get('VIDEO_DIR'))

    print('Getting frames...')
    extract_frames(video_dir, config)
    print('Detecting people and saving BBs...')
    detect_people(video_dir, config)
    print('Extracting image patches...')
    extract_patches(video_dir, config)
