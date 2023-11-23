#!/usr/bin/env python3

import os, shutil, smach, rospy, rospkg

from deepface import DeepFace

name1 = "/together2/3.jpg"
name2 = "/lili2/0.jpg"

DIR_PATH = os.path.join(rospkg.RosPack().get_path('lasr_vision_face_recognition'), 'dataset')
def test():
    res = DeepFace.verify(img1_path=DIR_PATH+name1, img2_path=DIR_PATH+name2, enforce_detection=False, model_name="Facenet512")
    print(res)
    # objs = DeepFace.analyze(img_path=DIR_PATH+"/juan/0.jpg",
    #                         actions=['age', 'gender', 'race', 'emotion']
    #                         )
    # print(objs)
    print(res['facial_areas']['img1'])
    facial= res['facial_areas']['img1']
    visualise(facial['x'], facial['y'], facial['x'] + facial['w'], facial['y'] + facial['h'])

def visualise(x1, y1, x2, y2):
    import cv2

    image = cv2.imread(DIR_PATH+ name1)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(DIR_PATH + name1 + "_image_with_box.jpg", image)


if __name__ == '__main__':
    rospy.init_node('test')
    test()

