import cv2
import os
import random
import pandas as pd
from pathlib import Path
import json

def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


# Instructions:
'''
Look at the frame. If there are no humans in the frame, and no ID annotations, click 'x' for no persons.
If there are people in the frame, check whether for each person, the ID is correct.
If any IDs are incorrect or missing, note it as incorrect. 
An incorrect result is a result where either someone who is visible is not detected, or where someone
has been assigned the incorrect ID. The result has to be perfect to get a 'correct' label.
'''


video_paths = [
    '/media/Storage_SSD3/BART/23-4-2024_#15_INDIVIDUAL_[16]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/27-3-2024_#15_INDIVIDUAL_[15]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/16-11-2023_#2_INDIVIDUAL_[16]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/10-10-2023_#1003_INDIVIDUAL_[2]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/10-1-2024_#6_INDIVIDUAL_[15]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/20-2-2024_#15_INDIVIDUAL_[2]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/27-2-2024_#10_INDIVIDUAL_[16]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/16-11-2023_#3_INDIVIDUAL_[18]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/14-3-2024_#15_INDIVIDUAL_[18]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/11-1-2024_#9_INDIVIDUAL_[18]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/11-1-2024_#7_INDIVIDUAL_[14]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/20-2-2024_#12_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',    # 5
    '/media/Storage_SSD3/BART/19-3-2024_#16_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',    # 4
    '/media/Storage_SSD3/BART/9-11-2023_#1_GROUP_[3-1]/camera_a_with_ids_openpose.mp4',     # 5 y
    '/media/Storage_SSD3/BART/15-5-2024_#19_INDIVIDUAL_[12]/camera_a_with_ids_openpose.mp4',    # 3
    '/media/Storage_SSD3/BART/12-6-2024_#22_INDIVIDUAL_[12]/camera_a_with_ids_openpose.mp4',  # 3
    '/media/Storage_SSD3/BART/12-12-2023_#6_GROUP_[10-17]/camera_a_with_ids_openpose.mp4',  # 4
    '/media/Storage_SSD3/BART/29-5-2024_#21_INDIVIDUAL_[12]/camera_a_with_ids_openpose.mp4',  # 3
    '/media/Storage_SSD3/BART/26-6-2024_#23_INDIVIDUAL_[12]/camera_a_with_ids_openpose.mp4',  # 3
    '/media/Storage_SSD3/BART/22-5-2024_#20_INDIVIDUAL_[12]/camera_a_with_ids_openpose.mp4',  # 3
    '/media/Storage_SSD3/BART/12-3-2024_#15_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 3
    '/media/Storage_SSD3/BART/16-4-2024_#18_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 4
    '/media/Storage_SSD3/BART/14-5-2024_#21_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 4
    '/media/Storage_SSD3/BART/8-5-2024_#18_INDIVIDUAL_[12]/camera_a_with_ids_openpose.mp4',  # 3
    '/media/Storage_SSD3/BART/12-3-2024_#15_GROUP_[10-17]/camera_a_with_ids_openpose.mp4',  # 4
    '/media/Storage_SSD3/BART/21-5-2024_#22_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 4
    '/media/Storage_SSD3/BART/28-11-2023_#5_GROUP_[20-9]/camera_a_with_ids_openpose.mp4',  # 5 y
    '/media/Storage_SSD3/BART/31-10-2023_#1_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 5 y
    '/media/Storage_SSD3/BART/7-11-2023_#2_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 5 y
    '/media/Storage_SSD3/BART/15-5-2024_#20_INDIVIDUAL_[15]/camera_a_with_ids_openpose.mp4',  # 2
    '/media/Storage_SSD3/BART/21-11-2023_#4_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 5 y
    '/media/Storage_SSD3/BART/19-12-2023_#7_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 3
	'/media/Storage_SSD3/BART/23-4-2024_#19_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 4
	'/media/Storage_SSD3/BART/5-12-2023_#6_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 5 y
	'/media/Storage_SSD3/BART/5-3-2024_#14_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 5 y
	'/media/Storage_SSD3/BART/28-5-2024_#23_GROUP_[9-20]/camera_a_with_ids_openpose.mp4',  # 4
	'/media/Storage_SSD3/BART/21-12-2023_#5_GROUP_[1-3]/camera_a_with_ids_openpose.mp4',  # 6
]


def select_random_frame(video_path, existing_samples):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        random_frame_number = random.randint(0, total_frames - 1)
        sample_identifier = (video_path, random_frame_number)

        # Check if the sample has already been selected
        if sample_identifier not in existing_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
            ret, frame = cap.read()
            cap.release()
            return frame, random_frame_number, sample_identifier
        else:
            print('Already checked this frame! Sampling another.')

def save_frame(frame, save_path, test_number):
    filename = f"test_{test_number}.jpg"
    full_path = os.path.join(save_path, filename)
    cv2.imwrite(full_path, frame)
    return full_path


def get_results_already_checked(csv_file):
    # Check if the CSV file already exists and determine the starting test number if results already exist for this video
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            last_test_number = df['test_number'].max()
            # Create a set of existing samples (video path, frame number)
            existing_samples = set(zip(df['video_path'], df['frame_number']))
        else:
            last_test_number = 0
            existing_samples = set()
    else:
        df = pd.DataFrame(columns=['test_number', 'response', 'saved_frame_path', 'video_path', 'frame_number'])
        last_test_number = 0
        existing_samples = set()

    test_number = last_test_number + 1

    return test_number, existing_samples, df


def main(video_paths):

    while True:
        # Randomly select a video and a frame, ensuring it's not a duplicate
        video_path = random.choice(video_paths)
        video_folder = os.path.dirname(video_path)
        config_file = os.path.join(video_folder, 'config_reid.json')

        config = load_config(config_file=config_file)
        print(config)
        max_n = config.get('MAX_N', 2)  # Default to 2 if not found
        save_path = f'test_frames_N={str(int(max_n))}'
        Path(save_path).mkdir(exist_ok=True)
        csv_file = os.path.join(save_path, 'results.csv')

        test_number, existing_samples, df = get_results_already_checked(csv_file)

        frame, frame_number, sample_identifier = select_random_frame(video_path, existing_samples)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Wait for user input
        key = cv2.waitKey(0)

        if key == ord('y'):
            response = 'correct'
        elif key == ord('n'):
            response = 'incorrect'
        elif key == ord('x'):
            response = 'no person'
        else:
            print("Exiting...")
            break

        # Save the frame
        saved_frame_path = save_frame(frame, save_path, test_number)

        # Add the new sample to the existing_samples set
        existing_samples.add(sample_identifier)

        # Create a new row for the results
        new_row = pd.DataFrame({
            'test_number': [test_number],
            'response': [response],
            'saved_frame_path': [saved_frame_path],
            'video_path': [video_path],
            'frame_number': [frame_number]
        })

        # Append the new row to the DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

        # Save the DataFrame back to CSV
        df.to_csv(csv_file, index=False)

        print(f"Test {test_number}: Saved frame {frame_number} from {video_path} as {response} at {saved_frame_path}")

        # Increment the test number for the next iteration
        test_number += 1

    cv2.destroyAllWindows()

main(video_paths)