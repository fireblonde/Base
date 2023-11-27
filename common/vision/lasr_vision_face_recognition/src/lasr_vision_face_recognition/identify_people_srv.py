#!/usr/bin/env python3

import os, shutil, smach, rospy, rospkg
import random

import numpy as np
from deepface.DeepFace import verify
from lasr_vision_face_recognition.srv import Identify, IdentifyResponse
from cv_bridge3 import CvBridge
from lasr_vision_msgs.msg import Detection
import cv2

name1 = "/lili/0.jpg"

class IdentifyPeople:
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'], input_keys=['name'])

        self.path = os.path.join(rospkg.RosPack().get_path('lasr_vision_face_recognition'), 'dataset')
        self.bridge = CvBridge()
        self.names = self.get_all_known_names()

    def visualise(self, x1, y1, x2, y2, image, name):
        # image = cv2.imread(self.path + name1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.getcwd() + "/image_with_box_" + name + ".jpg", image)


    def get_all_known_names(self):
        names = {}
        for filename in os.listdir(self.path):
            if os.path.isdir(os.path.join(self.path, filename)):
                names[filename] = os.path.join(self.path, filename) + "/0.jpg"
        return names


    def identify_face(self, im):
        # bbox of person + new background
        height, width, _ = im.shape
        new_height, new_width = 480, 640
        background = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        center_x = background.shape[1] // 2
        center_y = background.shape[0] // 2
        top_left_y = center_y - height // 2
        top_left_x = center_x - width // 2
        background[top_left_y:top_left_y + height, top_left_x:top_left_x + width] = im
        r = str(random.randint(0, 1000))
        name_im = "temp" + r + ".jpg"
        dataset_im = cv2.imwrite(name_im, background)

        # get the image path
        dataset_im = os.getcwd() + '/' + name_im

        detection = Detection()
        # go through dataset and match if any
        for name, filename in self.names.items():
            # check file
            if filename.endswith(".jpg"):
                # verify the given cutout with the known faces
                res = verify(img1_path=dataset_im, img2_path=filename, enforce_detection=False, model_name="Facenet512")
                print()
                print(f'the filename {filename}')
                print(f'the im {dataset_im}')
                print(res)
                print()
                print(res['verified'])
                if res['facial_areas'] and res['verified']:
                    # name the person
                    detection.name = name
                    detection.confidence = 1 - res['distance']
                    _res = res['facial_areas']['img1']
                    detection.xywh = _res['x'], _res['y'], _res['x'] + _res['w'], _res['y'] + _res['h']
                    self.visualise(_res['x'], _res['y'], _res['x'] + _res['w'], _res['y'] + _res['h'], background, name_im)
                    return detection

        # if no match, return empty detection
        # if detection.name == "":
        #     detection.name = "Unidentified"
        #     detection.confidence = 0.0
        #     detection.xywh = 0, 0, 0, 0

        print("the detected person is ")
        print(detection)
        return detection


    # the req is the orig_im and the detected_people
    def __call__(self, req):
        # transfer to cv img
        im = self.bridge.imgmsg_to_cv2(req.img_msg)
        cv2.imwrite("tmp.jpg", im)
        identified_people = []
        for person in req.detected_people:
            # get the bounding box of one person from the image
            contours = np.array(person.xyseg).reshape(-1, 2)
            x1, y1 = contours.min(axis=0)
            x2, y2 = contours.max(axis=0)
            self.visualise(x1, y1, x2, y2, im, "tmp")
            _im = im[y1:y2, x1:x2]

            # identify the face based on dataset
            one_face = self.identify_face(_im)
            if one_face.name:
                identified_people.append(one_face)
            else:
                print("No face identified")
                # have an empty detection

        print('all identified people are ************* ')
        print(identified_people)
        if len(identified_people) > 0:
            return IdentifyResponse(identified_people, True)
        return IdentifyResponse(identified_people, False)



if __name__ == "__main__":
    rospy.init_node("lasr_vision_face_recognition_identify_people")
    ip = IdentifyPeople()
    rospy.Service("/lasr_vision/identify_people", Identify, ip)
    rospy.spin()


