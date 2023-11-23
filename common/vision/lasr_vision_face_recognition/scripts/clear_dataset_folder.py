#!/usr/bin/env python3

import os
import shutil
import rospkg

# delete everything from the dataset folder

DIR_PATH = os.path.join(rospkg.RosPack().get_path('lasr_vision_face_recognition'), 'dataset')

print("Clearing dataset folder")
for folder in os.listdir(DIR_PATH):
    shutil.rmtree(os.path.join(DIR_PATH, folder))


print("Dataset folder is cleared")
