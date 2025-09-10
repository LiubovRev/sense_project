import pandas as pd
import numpy as np
list_to_exclude = []  # Exclusion list, if any


def compute_accuracy(csv_file, list_to_exclude):
	# Read the CSV file
	df = pd.read_csv(csv_file)

	# Extract 'N' from the folder name
	N = int(csv_file.split('test_frames_N=')[1].split('/')[0])

	total_all_cases = len(df)

	# Filter out rows where the response is 'no person'
	df_filtered = df[df['response'] != 'no person']

	# Exclude video paths in the exclusion list
	df_filtered = df_filtered[
		~df_filtered['video_path'].apply(lambda x: any(exclude_item in x for exclude_item in list_to_exclude))]

	# Calculate total cases
	total_cases = len(df_filtered)

	# Count correct cases
	correct_cases = len(df_filtered[df_filtered['response'] == 'correct'])

	# Calculate overall accuracy
	accuracy = correct_cases / total_cases if total_cases > 0 else 0.0

	# Print summary of results
	print(f"Total Cases (including 'no person'): {total_all_cases}")
	print(f"Total Cases (ignoring 'no person'): {total_cases}")
	print(f"Correct Cases: {correct_cases}")
	print(f"Incorrect Cases: {total_cases - correct_cases}")
	print(f"Accuracy: {accuracy:.2%}")

	# Group by video_path, calculate accuracy per video, and count the number of samples
	per_video_accuracy = (
		df_filtered.groupby('video_path').agg(
			accuracy=('response', lambda x: (x == 'correct').mean()),
			num_samples=('response', 'count')  # Count the number of samples per video
		).reset_index()
	)

	# Add 'N' column for the number of individuals in the video
	per_video_accuracy['N'] = N

	return per_video_accuracy


# List of CSV files
csv_files = [
	'/media/Storage_SSD3/BART_EVAL/test_frames_N=2/results.csv',
	'/media/Storage_SSD3/BART_EVAL/test_frames_N=3/results.csv',
	'/media/Storage_SSD3/BART_EVAL/test_frames_N=4/results.csv',
	'/media/Storage_SSD3/BART_EVAL/test_frames_N=5/results.csv'
]

# DataFrame to hold per-video accuracies
all_video_accuracies = pd.DataFrame()

# Loop through each file and compute accuracy
for file in csv_files:
	print(f'---------------------{file}------------------------')
	video_accuracies = compute_accuracy(file, list_to_exclude)

	# Append per-video accuracies to the main DataFrame
	all_video_accuracies = pd.concat([all_video_accuracies, video_accuracies], ignore_index=True)

# Save the per-video accuracies to a CSV file
all_video_accuracies.to_csv('per_video_accuracies_with_samples.csv', index=False)
print(len(np.unique(all_video_accuracies[all_video_accuracies.N == 2]['video_path'].values)),np.unique(all_video_accuracies[all_video_accuracies.N == 2]['video_path'].values))
print(len(np.unique(all_video_accuracies[all_video_accuracies.N == 3]['video_path'].values)),np.unique(all_video_accuracies[all_video_accuracies.N == 3]['video_path'].values))
print(len(np.unique(all_video_accuracies[all_video_accuracies.N == 4]['video_path'].values)),np.unique(all_video_accuracies[all_video_accuracies.N == 4]['video_path'].values))
print(len(np.unique(all_video_accuracies[all_video_accuracies.N == 5]['video_path'].values)),np.unique(all_video_accuracies[all_video_accuracies.N == 5]['video_path'].values))
print("Per-video accuracies with sample counts have been saved to 'per_video_accuracies_with_samples.csv'.")
