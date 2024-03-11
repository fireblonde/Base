#!/usr/bin/env python3

import os
import math
import smach
import rospy
from lasr_vision_msgs.srv import YoloDetection
from typing import List, Union
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Pose, PointStamped
import tf2_ros
from tf2_ros import TransformListener, Buffer


def calculate_distance(position1, position2):
    dx = position2.x - position1.x
    dy = position2.y - position1.y
    dz = position2.z - position1.z
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def get_center_bbox(xywh):
    x, y, w, h = xywh
    p = PointStamped()
    p.x = x + w / 2
    p.y = y + h / 2
    p.z = 0  # assuming same plane
    return p


def calculate_relative_position(object1, object2):
    center1_x = object1.x + object1.x / 2
    center1_y = object1.y + object1.y / 2
    center2_x = object2.x + object2.x / 2
    center2_y = object2.y + object2.y / 2

    # Calculate the relative position
    rel_pose = Pose()
    rel_pose.position.x = center2_x - center1_x
    rel_pose.position.y = center2_y - center1_y
    # Assuming objects are on the same plane, so the relative z position is 0
    rel_pose.position.z = 0

    return rel_pose


# heaviest/ lightest
def determine_weight(object1, object2):
    # look form a yaml file for the object names
    object_weight = rospy.get_param('object_weight')
    if object_weight[object1] > object_weight[object2]:
        weight = 'heavier'
    elif object_weight[object1] < object_weight[object2]:
        weight = 'lighter'
    else:
        weight = 'equal'
    return weight


# biggest / smallest
def determine_size(object1, object2):
    # look form a yaml file for the object names
    object_size = rospy.get_param('object_size')
    if object_size[object1] > object_size[object2]:
        size = 'bigger'
    elif object_size[object1] < object_size[object2]:
        size = 'smaller'
    else:
        size = 'equal'
    return size


# left of / right of
def determine_direction(object1, object2):
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
            known_objects: Union[List[str], None] = ['cup', 'bottle', 'banana'],
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

    #     self.objects = {'X': [], 'Y': [], 'Z': []}
    #     self.tf_buffer = Buffer()
    #     self.tf_listener = TransformListener(self.tf_buffer)
    #
    # def insert_object(self, bbox, frame_id):
    #     object_point = get_center_bbox(bbox)
    #     try:
    #         transform = self.tf_buffer.lookup_transform('map', frame_id, rospy.Time())
    #         transformed_point = self.tf_buffer.transform(object_point, 'map', timeout=rospy.Duration(1.0))
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
    #         rospy.logerr("Transform error: {}".format(ex))
    #         return
    #
    #     self.objects['X'].append((transformed_point.point.x, object_point))
    #     self.objects['Y'].append((transformed_point.point.y, object_point))
    #     self.objects['Z'].append((transformed_point.point.z, object_point))
    #
    #     self.objects['X'].sort(key=lambda x: x[0])
    #     self.objects['Y'].sort(key=lambda x: x[0])
    #     self.objects['Z'].sort(key=lambda x: x[0])
    #
    # def get_objects(self):
    #     return self.objects

    def execute(self, userdata):
        img_msg = rospy.wait_for_message(self.image_topic, Image)
        try:
            result = self.yolo(img_msg, "yolov8n.pt", self.confidence, self.nms)

            # get the known objects
            objects_poses = []
            for i, detection in enumerate(result.detected_objects):
                if detection.name in self.known_objects:
                    unique_object_name = f'{detection.name}_{i}'
                    print(unique_object_name)
                    objects_poses.append((unique_object_name, get_center_bbox(detection.xywh)))

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


if __name__ == '__main__':
    rospy.init_node('object_semantic_info')
    sm = smach.StateMachine(outcomes=['succeeded', 'failed'])
    with sm:
        smach.StateMachine.add('GET_OBJECT_SEMANTIC_INFO', ObjectSemanticInfo(),
                               transitions={'succeeded': 'succeeded', 'failed': 'failed'})
    outcome = sm.execute()
    print(outcome)
    print(sm.userdata.object_semantic_info)
    rospy.spin()
