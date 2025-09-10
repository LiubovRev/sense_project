
'''This code samples from two hand specified examples which are labelled well vs labelled badly, and provides an interface
for providing details on the assignment of IDs. The output is two csvs for the good and bad examples which can then be used to create a confusion matrix.'''

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Paths for good and bad examples
bad_example = '12-6-2024_#22_INDIVIDUAL_[12]'
good_example = '19-3-2024_#16_GROUP_[9-20]'
good_sample_frames_dir = 'test_frames_N=4'
bad_sample_frames_dir = 'test_frames_N=3'
good_results_key_fn = os.path.join(good_sample_frames_dir, 'results.csv')
bad_results_key_fn = os.path.join(bad_sample_frames_dir, 'results.csv')

# CSV files to save labeling results
saved_results_good_fn = 'labeling_results_good.csv'
saved_results_bad_fn = 'labeling_results_bad.csv'

# Load results from CSV (use correct file paths for results key)
good_results_key = pd.read_csv(good_results_key_fn)
bad_results_key = pd.read_csv(bad_results_key_fn)

# Subset results for both the good and bad examples
good_results = good_results_key[good_results_key['video_path'].str.contains(good_example, regex=False)]
bad_results = bad_results_key[bad_results_key['video_path'].str.contains(bad_example, regex=False)]

# Define individuals (A, B, C, D)
individuals = ['A', 'B', 'C', 'D']

# Initialize a 4x4 confusion matrix for good and bad examples
confusion_matrix_result_good = np.zeros((4, 4), dtype=int)
confusion_matrix_result_bad = np.zeros((3, 3), dtype=int)


def initialize_example(label_type):
    if label_type == "good":
        individuals = ['A', 'B', 'C', 'D']
        confusion_matrix = np.zeros((4, 4), dtype=int)
    elif label_type == "bad":
        individuals = ['A', 'B', 'C']
        confusion_matrix = np.zeros((3, 3), dtype=int)
    return individuals, confusion_matrix


# Define a function to map individual names to indices for the confusion matrix
def get_index(individual, individuals):
    return individuals.index(individual)

# Load existing results for good and bad examples if they exist
if os.path.exists(saved_results_good_fn):
    saved_results_good = pd.read_csv(saved_results_good_fn)
else:
    saved_results_good = pd.DataFrame(columns=['frame_number', 'true_label_A', 'true_label_B', 'true_label_C', 'true_label_D'])

if os.path.exists(saved_results_bad_fn):
    saved_results_bad = pd.read_csv(saved_results_bad_fn)
else:
    saved_results_bad = pd.DataFrame(columns=['frame_number', 'true_label_A', 'true_label_B', 'true_label_C', 'true_label_D'])

# Function to check if a frame has already been processed
def is_frame_already_labeled(frame_number, saved_results):
    return frame_number in saved_results['frame_number'].values

# Function to save results after each frame is processed
def save_results(frame_number, true_labels, saved_results, save_path, individuals):
    # Adjust the columns depending on how many individuals there are
    columns = ['frame_number'] + [f'true_label_{person}' for person in individuals]

    # Check if frame has already been processed, if not, append new row
    if not is_frame_already_labeled(frame_number, saved_results):
        new_row = pd.DataFrame([dict(zip(columns, [frame_number] + true_labels))])

        # Use pd.concat to append the new row to the DataFrame
        saved_results = pd.concat([saved_results, new_row], ignore_index=True)

        # Save the updated DataFrame to the CSV, ensuring previous results are retained
        saved_results.to_csv(save_path, index=False)
    return saved_results

# Function to process frames, display images, and update the confusion matrix
# Function to process frames, display images, and update the confusion matrix
def process_results(results, confusion_matrix_result, label_type, saved_results, save_path):
    individuals, confusion_matrix_result = initialize_example(label_type)
    print(f"\nProcessing {label_type} example:")

    for index, row in results.iterrows():
        frame_number = row['frame_number']

        # Check if this frame has already been labeled
        if is_frame_already_labeled(frame_number, saved_results):
            print(f"Frame {frame_number} has already been labeled. Skipping...")
            continue

        # Load and display the frame
        frame_path = row['saved_frame_path']
        img = cv2.imread(frame_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying correctly
        plt.title(f"Frame {frame_number}")
        plt.show()

        # Ask if the user wants to skip this frame
        skip_frame = input("Type 'skip' to skip this frame, or press Enter to continue: ").strip().lower()
        if skip_frame == 'skip':
            print(f"Skipping frame {frame_number}...")
            continue

        # Manually input the true labels for individuals after reviewing the frame
        true_labels = []
        for person in individuals:
            true_label = input(f"Enter true label for {person} ({', '.join(individuals)}): ")
            true_labels.append(true_label)

        # The detected labels will come from your detection results
        # Replace this with the actual detected labels from your data for the frame
        detected_labels = individuals  # This is a placeholder, update it with actual detected labels

        # Update the confusion matrix
        for i in range(len(individuals)):
            true_index = get_index(true_labels[i], individuals)
            detected_index = get_index(detected_labels[i], individuals)
            confusion_matrix_result[true_index, detected_index] += 1

        # Save the results after each frame is processed and update saved_results in memory
        saved_results = save_results(frame_number, true_labels, saved_results, save_path, individuals)

    return confusion_matrix_result

# Process good example and save results in 'labeling_results_good.csv'
# confusion_matrix_result_good = process_results(
#     good_results,
#     confusion_matrix_result_good,
#     "good",
#     saved_results_good,
#     saved_results_good_fn
# )

# Process bad example and save results in 'labeling_results_bad.csv'
confusion_matrix_result_bad = process_results(
    bad_results,
    confusion_matrix_result_bad,
    "bad",
    saved_results_bad,
    saved_results_bad_fn
)

