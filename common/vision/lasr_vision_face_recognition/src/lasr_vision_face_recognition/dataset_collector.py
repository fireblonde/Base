#!/usr/bin/env python3

import os, shutil, smach, rospy, rospkg
import random, string
from sensor_msgs.msg import Image

from cv_bridge3 import CvBridge
import cv2



IMAGE_NUMBER = rospy.get_param('/dataset_collector/image_number')
SLEEP_TIME = rospy.get_param('/dataset_collector/sleep_time')
MAX_DURATION = IMAGE_NUMBER * SLEEP_TIME + 2
DIR_PATH = os.path.join(rospkg.RosPack().get_path('lasr_vision_face_recognition'), 'dataset')
DEBUG = rospy.get_param('/dataset_collector/debug') or rospy.get_param('~debug', False)


class DatasetCollector():
    def __init__(self, topic):
        self.path = ''
        self.images_taken = 0
        self.semaphore = False
        self.bridge = CvBridge()
        self.topic = topic

        self.sub = rospy.Subscriber(self.topic, Image, self.img_callback)

    def img_callback(self, msg):
        if self.semaphore:
            if rospy.Time.now().to_sec() - self.last_time.to_sec() >= SLEEP_TIME and self.images_taken < IMAGE_NUMBER:
                cv_image = self.bridge.imgmsg_to_cv2_np(msg)
                cv2.imwrite(os.path.join(self.path, f'{self.images_taken}.jpg'), cv_image)
                self.images_taken += 1
                if DEBUG:
                    print("*" * 20, " - IMAGE IS TAKEN ", self.images_taken, "*" * 20)
                self.last_time = rospy.Time.now()
            elif self.images_taken >= IMAGE_NUMBER or rospy.Time.now().to_sec() - self.start_time.to_sec() > MAX_DURATION:
                self.semaphore = False

    def collect(self, name=''):
        rand = ''.join(random.choice(string.ascii_lowercase) for _ in range(7)) if not name else name
        dataset_path = self.path = os.path.join(DIR_PATH, rand)

        # remove the folder if it already exists
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        os.mkdir(self.path)

        self.semaphore = True
        self.start_time = self.last_time = rospy.Time.now()
        while self.semaphore:
            pass
        self.images_taken = 0
        return dataset_path, rand


if __name__ == '__main__':
    rospy.init_node('dataset_collector')
    # topic = '/xtion/rgb/image_raw'
    topic = '/usb_cam/image_raw'
    dc = DatasetCollector(topic)
    outcome = dc.collect('nicole')
    rospy.loginfo('\nI have completed execution with path: ')
    rospy.loginfo(outcome)
