import pandas as pd
import numpy as np
import json
import tarfile
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
maskdir_path = Path('/home/liubov/Bureau/new/MaskDir')

#frame_number	original_id	updated_id

chunk_folders = list(maskdir_path.glob('*'))
chunk_folders.sort()
print(chunk_folders)

result = []
total_frames = 0
n_frames = 1800
for i, folder in enumerate(chunk_folders[:-1]):
    videos = list(folder.glob("*"))
    videos.sort()
    if len(videos)==0:
        total_frames += n_frames

    for video_path in videos:

        cap = cv2.VideoCapture(video_path)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if video_path == videos[0]:
            total_frames += n_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames-1)
        ret, frame = cap.read()
        next_videos = list(chunk_folders[i+1].glob('*'))
        if len(next_videos) == 0:
            continue
        next_videos.sort()
        diff_frames = []
        for video in next_videos:
            cap2 = cv2.VideoCapture(video)
            ret2, frame2 = cap2.read()
            diff_frames.append(((frame - frame2)**2).sum())
        min_index = np.array(diff_frames).argmin()
        closest_video_path = next_videos[min_index]
        result.append(dict(frame_number=int(total_frames), original_id=int(video_path.stem), updated_id=int(closest_video_path.stem)))

df = pd.DataFrame(result)
df.to_csv(maskdir_path.parent/'first_id_mapping_camera_a.csv')
