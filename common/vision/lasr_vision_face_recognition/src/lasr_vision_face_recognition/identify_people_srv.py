#!/usr/bin/env python3

import os, shutil, smach, rospy, rospkg

import numpy as np
from deepface.DeepFace import verify
from lasr_vision_face_recognition.srv import Identify, IdentifyResponse
from cv_bridge3 import CvBridge
from lasr_vision_msgs.msg import Detection
import cv2

name1 = "/together2/3.jpg"
name2 = "/lili2/0.jpg"

class IdentifyPeople:
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'], input_keys=['name'])

        self.path = os.path.join(rospkg.RosPack().get_path('lasr_vision_face_recognition'), 'dataset')
        self.bridge = CvBridge()
        self.names = self.get_all_known_names()

    def visualise(self, x1, y1, x2, y2, image):
        # image = cv2.imread(self.path + name1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.getcwd() + "/image_with_box.jpg", image)


    def get_all_known_names(self):
        names = {}
        for filename in os.listdir(self.path):
            if os.path.isdir(os.path.join(self.path, filename)):
                names[filename] = os.path.join(self.path, filename) + "/0.jpg"
        return names


    def identify_faces(self, im):
        dataset_im = cv2.imwrite("temp.jpg", im)
        # get the image path
        dataset_im = os.getcwd() + "/temp.jpg"
        identified_people = []
        for name, filename in self.names.items():
            detection = Detection()
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
                    identified_people.append(detection)

        print(identified_people)
        return identified_people

    # the req is the orig_im and the detected_people
    def __call__(self, req):
        # transfer to cv img
        im = self.bridge.imgmsg_to_cv2(req.img_msg)
        cv2.imwrite("tmp.jpg", im)
        for person in req.detected_people:
            print("---")
            print(person.xywh)
            print(person.xyseg)
            contours = np.array(person.xyseg).reshape(-1, 2)
            x1, y1 = contours.min(axis=0)
            x2, y2 = contours.max(axis=0)
            print(x1, y1, x2, y2)
            # get the bbox from orig image
            # x1, y1, x2, y2 = person.xywh[0], person.xywh[1], person.xywh[2], person.xywh[3]
            # x1, y1, x2, y2 = person.xywh[0], person.xywh[1], person.xywh[2], person.xywh[3]
            # x1, y1, x2, y2 = person.xywh[0], person.xywh[1], person.xywh[0] + person.xywh[3], person.xywh[0] + person.xywh[2]
            # x1, y1, x2, y2 = person.xywh[0], person.xywh[1], person.xywh[0] + person.xywh[2], person.xywh[0] + person.xywh[3]
            # cutout the body and face from orig image

            self.visualise(x1, y1, x2, y2, im)
            _im = im[y1:y2, x1:x2]
            # identify the face
            identified_people = self.identify_faces(_im)
            if identified_people:
                return IdentifyResponse(identified_people, True)



        return IdentifyResponse(identified_people, False)



if __name__ == "__main__":
    rospy.init_node("lasr_vision_face_recognition_identify_people")
    ip = IdentifyPeople()
    rospy.Service("/lasr_vision/identify_people", Identify, ip)
    rospy.spin()


