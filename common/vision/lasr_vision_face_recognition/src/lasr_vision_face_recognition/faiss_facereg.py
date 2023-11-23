#!/usr/bin/env python3

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import rospkg
import faiss

from deepface import DeepFace

# TODO: FACE RECOGNITION - as vectors usiing DeepFace
# /home/nicole/.deepface/weights/facenet_weights.h5
detector_backend = "mtcnn"

name = 'nicole'
DIR_PATH = os.path.join(rospkg.RosPack().get_path('lasr_vision_face_recognition'), 'dataset')


def find_brightest_image(name):
    max_brightness = 0
    # brightest_image = None
    brightest_image_path = None
    path = os.path.join(DIR_PATH, name)

    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path)
            brightness = image.mean()
            print(f"{filename} - {brightness}")

            if brightness > max_brightness:
                max_brightness = brightness
                # brightest_image = filename
                brightest_image_path = image_path

    return brightest_image_path


print(f"The brightest image is: {find_brightest_image(name)}")


def read_dataset_path(name):
    files = []
    path = os.path.join(DIR_PATH, name)
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(path, filename)
            files.append(file_path)

    return files


print(read_dataset_path(name))
#
# model = Facenet # Or FbDeepFace Or VGGFace Or OpenFace
# if(model==Facenet):
#   model = Facenet.loadModel()
#   dim = 128
#   ip_shape = 160
# if(model==FbDeepFace):
#   model = FbDeepFace.loadModel()
#   dim = 4096
#   ip_shape = 152
# if(model==VGGFace):
#   model = VGGFace.loadModel()
#   dim = 2622
#   ip_shape = 224
# if(model==OpenFace):
#   model = OpenFace.loadModel()
#   dim = 128
#   ip_shape = 96

Similarity_Metric = 1


def get_representatios_faiss(model_name):
    names = [folder for folder in os.listdir(DIR_PATH)]
    files = [find_brightest_image(name) for name in names]
    print(f'Files: {files}')
    print(f'Names: {names}')
    representations = []
    for img_path in files:
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        print(img_path)

        representation = []
        representation.append(img_path)
        representation.append(embedding)
        representations.append(representation)
    return representations


representations = get_representatios_faiss("DeepFace")


# # Faiss expect 2 dimensional matrix as float32 numpy array type.

def get_target_representation(representations, target_path, model_name):
    embeddings = [representations[i][1] for i in range(0, len(representations))]

    embeddings = np.array(embeddings, dtype='f')
    target_representation = DeepFace.represent(img_path=target_path, model_name=model_name, enforce_detection=False, detector_backend=detector_backend)[0][
        "embedding"]
    target_representation = np.array(target_representation, dtype='f')
    target_representation = np.expand_dims(target_representation, axis=0)
    return target_representation, embeddings


target, embeddings = get_target_representation(get_representatios_faiss("DeepFace"), DIR_PATH + '/nicole/2.jpg', "Facenet")
print(f'Target: {target}')
print(f'Embeddings: {len(embeddings)}')

def get_similarity(target, embeddings, metric):
    dimensions = 128  # FaceNet output is 128 dimensional vector
    if metric == 'euclidean':
        index = faiss.IndexFlatL2(dimensions)
        distances = np.linalg.norm(embeddings - target, axis=1)
    elif metric == 'cosine':
        distances = np.dot(embeddings, target.T).flatten()
        index = faiss.IndexFlatIP(dimensions)
        faiss.normalize_L2(embeddings)
    return distances, index

dist, index = get_similarity(target, embeddings, 'euclidean')

def save_index(index, path):
    # faiss.write_index(index, "vector.index")
    faiss.write_index(index, path)

def read_index(path):
    # index = faiss.read_index("vector.index")
    return faiss.read_index(path)

def search(index, target, k):
    distances, indices = index.search(target, k)
    return distances, indices

print('search')
distances, indices = search(index, target, 3)

for idx, neighbour in enumerate(indices[0]):
    neighbour_name = representations[neighbour][0]

    neighbour_img = DeepFace.extract_faces(
        img_path=neighbour_name,
        detector_backend=detector_backend
    )[0]["face"]

    print(f"{idx + 1}th closest image in {len(embeddings)} database")
    print(f"Image name: {neighbour_name}")
    print(f"Distance: {distances[0][idx]}")
    print(f"Target: {DIR_PATH + '/nicole/2.jpg'}")
    print()
    print()
    print()
    print()

