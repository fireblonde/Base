#!/usr/bin/env python3

import os
import rospkg

# rename the dataset folder

DIR_PATH = os.path.join(rospkg.RosPack().get_path('lasr_vision_face_recognition'), 'dataset')

print("\nFolders in dataset folder:")
for folder in os.listdir(DIR_PATH):
    print("-"*5, folder, "-"*5)

name_in = input("\nInput the name of the folder to rename:\n")
name_out = input("\nInput the new name of the folder: \n")


print("\nRenaming dataset folder\n")
def rename_folder(name_in, name_out):
    for folder in os.listdir(DIR_PATH):
        if folder == name_in:
            os.rename(os.path.join(DIR_PATH, folder), os.path.join(DIR_PATH, name_out))
            print(f"Renamed {folder} to {name_out}")
            return
    print(f"Folder {name_in} not found")


rename_folder(name_in, name_out)
