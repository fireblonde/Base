#!/usr/bin/env python3

import rospy
# import os
import smach
from sensor_msgs.msg import Image
from cv_bridge3 import CvBridge
# import cv2

from lasr_vision_msgs.srv import YoloDetection

class DetectObjects(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'], input_keys=['image_topic', 'filter'], output_keys=['img_msg', 'detections'])
        self.yolo = rospy.ServiceProxy('/yolov8/detect', YoloDetection)
        self.bridge = CvBridge()
    
    def execute(self, userdata):
        # im = cv2.imread(os.getcwd() + "/together.jpg")
        # img_msg = self.bridge.cv2_to_imgmsg(im)
        img_msg = rospy.wait_for_message(userdata.image_topic, Image)
        try:
            result = self.yolo(img_msg, "yolov8n-seg.pt", 0.5, 0.3)
            result.detected_objects = [det for det in result.detected_objects if det.name in userdata.filter]
            print(result.detected_objects)
            userdata.img_msg = img_msg
            userdata.detections = result
            return 'succeeded'
        except rospy.ServiceException as e:
            rospy.sleep(10)
            rospy.logwarn(f"Unable to perform inference. ({str(e)})")
            return 'failed'