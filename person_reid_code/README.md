# person_reidentification



# Instructions 06.10.2024

1. Navigate to the folder with the videos in and add a json file like config_reid.json:
```json
{
  "ROOT_DIR": "/media/matthewvowels/Storage_SSD3/BART/",
  "VIDEO_DIR": "15-5-2024_#20_INDIVIDUAL_[15]",
  "CAMERA_NAME": "camera_a.mkv",
  "COORD2d_DIR": "skeletons",
  "COORD3d_DIR": "skeletons3d",
  "PATCHES_DIR": "patches",
  "START_FRAME": 2000,
  "END_FRAME": 1000000,
  "VALIDATION_OUTPUT_DIR": "reid_validation",
  "FRAME_DIR": "frames",
  "BOUNDINGBOX_DIR": "bounding_boxes",
  "MAX_N": 2,
   "ID_TO_NAME_MAPPING": {
    "0": "Therapist1",
    "1": "Patient"
  }
}
```

2. Run this to extract the frames and use yolo to get bounding boxes and patches:
```bash
CONFIG_PATH="config_reid.json"
conda activate segment_anything
time python3 get_frames_and_detect.py --config "$CONFIG_PATH" 
```


3. Run this to mask the patches, embed them, and get the reidentification mapping:
```bash
conda activate mediapipe_torchreid 
time python3 skeleton_masks_reid.py --config "$CONFIG_PATH"
```


4. Run this to estimate the skeletons with openpose
```bash
docker run -it --user $(id -u):$(id -g) \
  -v /home/Psych_ML/person_reidentification:/home/Psych_ML/person_reidentification \
  -v /media/Storage_SSD3:/media/Storage_SSD3 \
  --net=host -e DISPLAY -e CONFIG_PATH="$CONFIG_PATH" --gpus all -e NVIDIA_VISIBLE_DEVICES=0 cwaffles/openpose

  
cd /home/Gith.....
./get_skeletons.sh -c "$CONFIG_PATH" 
```


4. Run this to estimate to visualize the results
```bash
time python3 visualize_mapping_openpose.py --config "$CONFIG_PATH"
```


5. The EVAL folder contains the following scripts:

```random_id_tester.py``` This provides an interface to manually assess the success of the assigned identities.

```evaluate_id_annotations.py``` This evaluates the results of the manual annotations for the re-identification.

```comparison_strong_sort.ipynb``` This takes the output of strong-sort and computes the number of estimated unique individuals.

```confusion_matrices_labelling.py``` This provides an interface for generating confusion matrix data for specific videos.

```generate_confusion_matrices.ipynb``` Using the data generated using the script above, this produces the confusion matrices themselves.

```plot_accuracies_per_video.ipynb``` is the script used to plot the main results for the re-identification.

```plot_trajectories.py``` generates the x-y-time coordinates of the participants for specific videos.




