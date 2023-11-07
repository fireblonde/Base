#!/usr/bin/env python3
import rospy
import smach
from lasr_vision_msgs.srv import YoloDetection


class DetectPeople(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'], input_keys=['img_msg'], output_keys=['people_detections'])
        self.yolo = rospy.ServiceProxy('/yolov8/detect', YoloDetection)

    def execute(self, userdata):
        try:
            result = self.yolo(userdata.img_msg, "yolov8n.pt", 0.5, 0.3)
            result.detected_objects = [det for det in result.detected_objects if det.name == "person"]
            userdata.people_detections = result
            return 'succeeded'
        except rospy.ServiceException as e:
            rospy.logwarn(f"Unable to perform inference. ({str(e)})")
            return 'failed'
