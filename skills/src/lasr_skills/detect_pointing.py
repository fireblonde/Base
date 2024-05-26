import rospy
import smach
from sensor_msgs.msg import Image

from lasr_vision_msgs.srv import PointingDirection


class DetectPointingDirection(smach.State):
    def __init__(self, image_topic: str = "/xtion/rgb/image_raw"):
        smach.State.__init__(
            self, outcomes=["succeeded", "failed"], output_keys=["pointing_direction"]
        )
        self._image_topic = image_topic
        self._pointing_service = rospy.ServiceProxy(
            "/pointing_detection_service", PointingDirection
        )
        self._pointing_service.wait_for_service()

    def execute(self, userdata):

        img_msg = rospy.wait_for_message(self._image_topic, Image)
        resp = self._pointing_service(img_msg)
        if resp.direction == "NONE" or resp.direction == "FORWARDS":
            return "failed"
        userdata.pointing_direction = resp.direction
        return "succeeded"