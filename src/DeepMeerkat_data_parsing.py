# %% 
import cv2
import datetime
import subprocess
import math
import matplotlib.pyplot as plt
import pandas as pd

# %% 
### Create VideoCapture object
videoFile = "/data/HummingbirdVideo/FH102_02.AVI"
cap = cv2.VideoCapture(videoFile)   # grab video file from path

# %% 
### from https://stackoverflow.com/a/61572332/7722773
def with_opencv(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return duration, frame_count

print(with_opencv('/data/HummingbirdVideo/FH102_02.AVI')) # this just reads from the file header, which might be wrong 

# counting the frames... 
total_frame = 0
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    total_frame += 1
print("total_frame = " + str(total_frame))

# %%
# calculate duration of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))
seconds = int(total_frame / fps)
video_time = str(datetime.timedelta(seconds=seconds))
print("duration in seconds:", seconds)
print('frame rate:', fps) # this also comes from the file header therefore might be wrong...
print("video time:", video_time)

### CAUTION: OpenCV is NOT accurate for counting frames exactly! See https://github.com/opencv/opencv/issues/15749#issuecomment-796512723

# # %%
# ### Alternative solution using external utility from FFmpeg - IN PROGRESS
# def with_ffprobe(filename):
#     import subprocess, json

#     result = subprocess.check_output(
#             f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
#             shell=True).decode()
#     fields = json.loads(result)['streams'][0]

#     duration = fields['tags']['DURATION']
#     fps      = eval(fields['r_frame_rate'])
#     return duration, fps

# print(with_ffprobe('/data/HummingbirdVideo/FH101_01.AVI'))

# %% 
### Extract frames from videos
frameRate = cap.get(5)  # get framerate (should be 1 fps)
count = 0
x = 1
while(cap.isOpened()):
    frameId = cap.get(1)    # current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = "frame%d.jpeg" % count; count += 1
        cv2.imwrite(filename, frame)
cap.release()
print("Done!")

# %%
### Show a few images
img = plt.imread("frame0.jpeg") # straight up read from filename
plt.imshow(img)

# %%
### Read in the labels file
labels = pd.read_csv("Weinstein2018MEE_ground_truth.csv")
labels.head
labels.dtypes

labels[labels["Video"] == "FH102_02"].Frame

# %%
labels[labels["Truth"] == "Positive"].any()
# labels[labels["Truth"] == "Positive"].Frame[140]

# # %%
# ### Print positive frames
# i = 1
# while i in labels[labels["Video"] == "FH102_02"].Frame: 
#     if labels["Truth"][i] == "Positive":
#         img = plt.imread("frame"+str(labels[labels["Video"] == "FH102_02"].Frame[i])+".jpeg")
#         plt.imshow(img)
#         print("cycle "+str(i))
#         wait = input("Press Enter to continue.")
#         i = i + 1 
#     i = i + 1
# print("All printed.")

# # not sure why it throws EOF error... 

# %%
plt.imshow(plt.imread("frame"+str(labels[labels["Video"] == "FH102_02"].Frame[133])+".jpeg"))
# %%
