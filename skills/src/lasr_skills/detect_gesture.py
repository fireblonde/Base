#!/usr/bin/env python3
import rospy
import smach
import rospkg
import os

from typing import Union, List
from lasr_vision_msgs.srv import YoloDetection
import cv2
from sensor_msgs.msg import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python import  ImageFormat
from cv_bridge import CvBridge

# path = os.path.join(rospkg.RosPack.get_path(name="common"), "helpers", "models", "gesture_recognizer.task")
path = "/home/nicole/robocup/base/src/Base/common/helpers/models/src/models/gesture_recognizer.task"

class DetectGesture(smach.State):

    def __init__(self,
                 filter: Union[List[str], None] = ["wave"],
                 image_topic: str = "/usb_cam/image_raw",
                 # image_topic: str = "/xtion/rgb/image_raw",
                 model: str = "yolov8n.pt",
                 confidence: float = 0.5,
                 nms: float = 0.3,
                 ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed"],
            output_keys=["gesture"],
        )
        self.model = model
        self.filter = filter if filter is not None else []
        self.image_topic = image_topic
        self.confidence = confidence
        self.nms = nms
        self.bridge = CvBridge()
        # self.yolo = rospy.ServiceProxy("/yolov8/detect", YoloDetection)
        # self.yolo.wait_for_service()

        self.base_options = python.BaseOptions(model_asset_path=path)
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def execute(self, userdata):
        img_msg = rospy.wait_for_message(self.image_topic, Image)
        try:
            #  use cv bridge to convert image to cv2
            msg = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            msg = cv2.cvtColor(msg, cv2.COLOR_BGR2RGB)
            # image = mp.Image(frame=msg)
            recognition_result = self.recognizer.recognize(msg, ImageFormat.IMAGEFORMAT_RGB)
            #  use mediapipe to detect gesture
            top_gesture = recognition_result.gestures[0][0]
            #  use mediapipe to detect hand landmarks
            hand_landmarks = recognition_result.hand_landmarks
            userdata.gesture = top_gesture
            return "succeeded"
        except rospy.ServiceException as e:
            rospy.logwarn(f"Unable to perform inference. ({str(e)})")
            return "failed"


if __name__ == "__main__":
    rospy.init_node("detect_gesture")
    detect_gesture = DetectGesture()
    ud = smach.UserData()
    detect_gesture.execute(ud)
    print(ud.gesture)
