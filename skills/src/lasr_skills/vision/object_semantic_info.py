#!/usr/bin/env python3

import os
import smach
import rospy
from lasr_vision_msgs.srv import YoloDetection
from typing import List, Union
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from typing import Tuple
from common.helpers.cv2_img import get_center_bbox, calculate_distance


def calculate_relative_position(object1, object2) -> Pose:
    # center1_x = object1.x + object1.x / 2
    # center1_y = object1.y + object1.y / 2
    # center2_x = object2.x + object2.x / 2
    # center2_y = object2.y + object2.y / 2
    #
    # # Calculate the relative position
    # rel_pose = Pose()
    # rel_pose.position.x = center2_x - center1_x
    # rel_pose.position.y = center2_y - center1_y
    # rel_pose.position.z = 0  # assuming same plane

    rel_pose = Pose()
    rel_pose.position.x = object2.x - object1.x
    rel_pose.position.y = object2.y - object1.y
    rel_pose.position.z = object2.z - object1.z

    return rel_pose


# left of / right of
def determine_direction(object1, object2) -> Tuple[str, str]:
    rel_pose = calculate_relative_position(object1, object2)

    direction_rl, direction_tb = '', ''
    if rel_pose.position.x > 0:
        direction_rl = 'left'
    elif rel_pose.position.x < 0:
        direction_rl = 'right'
    if rel_pose.position.y > 0:
        direction_tb = 'top'
    elif rel_pose.position.y < 0:
        direction_tb = 'bottom'

    return direction_rl, direction_tb


class ObjectSemanticInfo(smach.State):
    """
    State for reading an ObjectSemanticInfo message
    """

    def __init__(
            self,
            known_objects: Union[List[str], None] = ['cup', 'bottle', 'mouse', 'cell phone'],
            image_topic: str = "/xtion/rgb/image_raw",
            confidence: float = 0.5,
            nms: float = 0.3,
    ):
        smach.State.__init__(
            self,
            input_keys=[],
            outcomes=['succeeded', 'failed'],
            output_keys=['object_semantic_info'],
        )
        self.known_objects = known_objects if known_objects is not None else []
        self.image_topic = image_topic
        self.confidence = confidence
        self.nms = nms
        self.yolo = rospy.ServiceProxy("/yolov8/detect", YoloDetection)
        self.yolo.wait_for_service()

    def execute(self, userdata):
        img_msg = rospy.wait_for_message(self.image_topic, Image)
        try:
            result = self.yolo(img_msg, "yolov8n.pt", self.confidence, self.nms)

            # get the known objects
            objects_poses = []
            for i, detection in enumerate(result.detected_objects):
                if detection.name in self.known_objects:
                    # unique_object_name = f'{detection.name}_{i}'
                    # print(unique_object_name)
                    # objects_poses.append((unique_object_name, get_center_bbox(detection.xywh)))
                    objects_poses.append((detection.name, get_center_bbox(detection.xywh)))

            # get the object semantic info
            object_semantic_info = {}
            for object1_name, object1_position in objects_poses:
                for other_object_name, other_object_position in objects_poses[1:]:
                    if object1_name != other_object_name:
                        direction_rl, direction_tb = determine_direction(object1_position, other_object_position)
                        distance = calculate_distance(object1_position, other_object_position)
                        print(
                            f'{object1_name} is of direction: {direction_rl}, {direction_tb}, of {other_object_name} and is {distance} meters away')

                        if object1_name not in object_semantic_info:
                            object_semantic_info[object1_name] = {}
                        object_semantic_info[object1_name][other_object_name] = {'direction_rl': direction_rl,
                                                                                 'direction_tb': direction_tb,
                                                                                 'distance': distance}

            # save the semantic info
            # Userdata example
            # - object_semantic_info
            #  - cup_0
            #   - bottle_1
            #    - direction: top-right
            #    - distance: 1.5

            userdata.object_semantic_info = object_semantic_info
            return 'succeeded'
        except rospy.ServiceException as e:
            rospy.logwarn(f"Unable to perform inference. ({str(e)})")
            return 'failed'


class QueryObjectSemanticInfo(smach.State):
    """
    State for querying the object semantic info:

    Answer the question: Object is right/left of what object?
    """

    def __init__(self):
        smach.State.__init__(
            self,
            input_keys=['object_semantic_info', 'object1', 'direction'],
            outcomes=['succeeded', 'failed'],
            output_keys=['object'],
        )

    def execute(self, userdata):
        object_semantic_info = userdata.object_semantic_info
        object1 = userdata.object1
        direction = userdata.direction

        userdata.object = 'no object'
        if object1 in object_semantic_info:
            for other_object, info in object_semantic_info[object1].items():
                if info['direction_rl'] == direction or info['direction_tb'] == direction:
                    rospy.loginfo(f"output userdata {other_object}")
                    userdata.object = other_object
                    return 'succeeded'
        return 'failed'


if __name__ == '__main__':
    rospy.init_node('object_semantic_info')

    sm = smach.StateMachine(outcomes=['succeeded', 'failed'], input_keys=['object1', 'direction'],
                            output_keys=['object'])
    sm.userdata.object1 = 'cup'
    sm.userdata.direction = 'left'
    with sm:
        smach.StateMachine.add('GET_OBJECT_SEMANTIC_INFO', ObjectSemanticInfo(),
                               transitions={'succeeded': 'QUERY_OBJECT_SEMANTIC_INFO', 'failed': 'failed'},
                               remapping={'object_semantic_info': 'object_semantic_info'})
        smach.StateMachine.add('QUERY_OBJECT_SEMANTIC_INFO', QueryObjectSemanticInfo(),
                               transitions={'succeeded': 'succeeded', 'failed': 'failed'},
                               remapping={'object_semantic_info': 'object_semantic_info', 'object1': 'object1',
                                          'direction': 'direction', 'object': 'object'}
                               )
    outcome = sm.execute()
    print(sm.userdata.object)
    print(sm.userdata.object_semantic_info)
    print(outcome)
    rospy.spin()
