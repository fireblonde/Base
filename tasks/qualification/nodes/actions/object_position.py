#!/usr/bin/env python3

import typing

import rospy
from actionlib import SimpleActionServer, SimpleActionClient
from qualification.msg import GetObjectPositionAction, GetObjectPositionResult, ObjectPose, KeyValue
from lasr_vision_msgs.srv import YoloDetection
from common_math import pcl_msg_to_cv2, seg_to_centroid
from sensor_msgs.msg import PointCloud2, Image
from tf_module.srv import TfTransform, TfTransformRequest
from geometry_msgs.msg import PointStamped, Point, Pose
from std_msgs.msg import String
from cv_bridge3 import CvBridge
import numpy as np
import math


class ObjectPosition:
    def __init__(self):
        super().__init__()

        # wait for yolo
        self.yolo = rospy.ServiceProxy("/yolov8/detect", YoloDetection)
        self.tf = rospy.ServiceProxy("/tf_transform", TfTransform)
        self.bridge = CvBridge()

        self._as = SimpleActionServer(
            "object_position",
            GetObjectPositionAction,
            execute_cb=self.execute_cb,
            auto_start=False,
        )

        self.known_objects = ['cup', 'bottle']

        self._as.start()

    def estimate_pose(self, pcl_msg, detection):
        centroid_xyz = seg_to_centroid(pcl_msg, np.array(detection.xyseg))
        centroid = PointStamped()
        centroid.point = Point(*centroid_xyz)
        centroid.header = pcl_msg.header
        tf_req = TfTransformRequest()
        tf_req.target_frame = String("map")
        tf_req.point = centroid
        response = self.tf(tf_req)
        return response.target_point.point

    def calculate_relative_position(self, object1, object2):
        # Calculate the center of each bounding box
        # center1_x = object1[0] + object1[2] / 2
        # center1_y = object1[1] + object1[3] / 2
        # center2_x = object2[0] + object2[2] / 2
        # center2_y = object2[1] + object2[3] / 2
        print(object1)

        # do it with points instead
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

    def get_center_bbox(self, xywh):
        x, y, w, h = xywh
        p = Point()
        p.x = x + w/2
        p.y = y + h/2
        p.z = 0
        return p

    # def calculate_relative_position(object1, object2):
    #     rel_pose = Pose()
    #     rel_pose.position.x = object2.position.x - object1.position.x
    #     rel_pose.position.y = object2.position.y - object1.position.y
    #     rel_pose.position.z = object2.position.z - object1.position.z
    #     return rel_pose


    #
    def determine_direction(self, object1, object2):
        rel_pose = self.calculate_relative_position(object1, object2)

        direction_rl, direction_tb  = '', ''
        if rel_pose.position.x > 0:
            direction_rl = 'left'
        elif rel_pose.position.x < 0:
            direction_rl = 'right'
        if rel_pose.position.y > 0:
            direction_tb = 'top'
        elif rel_pose.position.y < 0:
            direction_tb = 'bottom'

        return direction_rl, direction_tb

        # if abs(rel_pose.position.x) > 0 and abs(rel_pose.position.y) > 0:
        #     direction = 'top-right'
        # elif abs(rel_pose.position.x) > 0 and abs(rel_pose.position.y) < 0:
        #     direction = 'bottom-right'
        # elif abs(rel_pose.position.x) < 0 and abs(rel_pose.position.y) > 0:
        #     direction = 'top-left'
        # elif abs(rel_pose.position.x) < 0 and abs(rel_pose.position.y) < 0:
        #     direction = 'bottom-left'
        # elif abs(rel_pose.position.x) > 0:
        #     direction = 'right'
        # elif abs(rel_pose.position.x) < 0:
        #     direction = 'left'
        # elif abs(rel_pose.position.y) > 0:
        #     direction = 'top'
        # elif abs(rel_pose.position.y) < 0:
        #     direction = 'bottom'
        # else:
        #     direction = 'i dunno'


    @staticmethod
    def calculate_distance(position1, position2):
        dx = position2.x - position1.x
        dy = position2.y - position1.y
        dz = position2.z - position1.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def execute_cb(self, goal):
        rospy.loginfo(goal)
        # pcl_msg = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2)
        # cv_im = pcl_msg_to_cv2(pcl_msg)
        # img_msg = self.bridge.cv2_to_imgmsg(cv_im)
        img_msg = rospy.wait_for_message("/xtion/rgb/image_raw",Image)
        yolo_res = self.yolo(img_msg, "yolov8n.pt", 0.5, 0.3)

        objects_poses = []
        print(yolo_res.detected_objects[0])
        for i, detection in enumerate(yolo_res.detected_objects):
            print(detection.name + "----------")
            if detection.name in goal.known_objects:
                unique_object_name = f'{detection.name}_{i}'
                print(unique_object_name)
                objects_poses.append((unique_object_name, self.get_center_bbox(detection.xywh)))

        result = GetObjectPositionResult()

        for object1_name, object1_position in objects_poses:

            closest_top = closest_bottom = closest_left = closest_right = None
            farthest_top = farthest_bottom = farthest_left = farthest_right = None
            min_top_distance = min_bottom_distance = min_left_distance = min_right_distance = float('inf')
            max_top_distance = max_bottom_distance = max_left_distance = max_right_distance = 0

            # For each object in the goal, find its relative position with respect to object1
            for other_object_name, other_object in objects_poses[1:]:
                if object1_name != other_object_name:
                    if other_object is not None:
                        direction_rl, direction_tb = self.determine_direction(object1_position, other_object)

                        distance = self.calculate_distance(object1_position, other_object)
                        print(f"Distance: {distance}, object1: {object1_name}, other: {other_object_name}")
                        print(f"Direction: {direction_rl}, {direction_tb}, object1: {object1_name}, other: {other_object_name}")

                        if direction_tb == 'top':
                            if distance < min_top_distance:
                                min_top_distance = distance
                                closest_top = other_object
                            if distance > max_top_distance:
                                max_top_distance = distance
                                farthest_top = other_object
                        elif direction_tb == 'bottom':
                            if distance < min_bottom_distance:
                                min_bottom_distance = distance
                                closest_bottom = other_object
                            if distance > max_bottom_distance:
                                max_bottom_distance = distance
                                farthest_bottom = other_object

                        if direction_rl == 'left':
                            if distance < min_left_distance:
                                min_left_distance = distance
                                closest_left = other_object
                            if distance > max_left_distance:
                                max_left_distance = distance
                                farthest_left = other_object
                        elif direction_rl == 'right':
                            if distance < min_right_distance:
                                min_right_distance = distance
                                closest_right = other_object
                            if distance > max_right_distance:
                                max_right_distance = distance
                                farthest_right = other_object

                    object1_pose = ObjectPose()
                    object1_pose.object = object1_name
                    object1_pose.positions = [
                        KeyValue(key='closest_top', value=Pose(position=closest_top), object_name=other_object_name),
                        KeyValue(key='farthest_top', value=Pose(position=farthest_top), object_name=other_object_name),
                        KeyValue(key='closest_bottom', value=Pose(position=closest_bottom), object_name=other_object_name),
                        KeyValue(key='farthest_bottom', value=Pose(position=farthest_bottom), object_name=other_object_name),
                        KeyValue(key='closest_left', value=Pose(position=closest_left), object_name=other_object_name),
                        KeyValue(key='farthest_left', value=Pose(position=farthest_left), object_name=other_object_name),
                        KeyValue(key='closest_right', value=Pose(position=closest_right), object_name=other_object_name),
                        KeyValue(key='farthest_right', value=Pose(position=farthest_right), object_name=other_object_name)
                    ]

                    result.object_positions.append(object1_pose)

            self._as.set_succeeded(result)




if __name__ == "__main__":
    rospy.init_node("detect_object_position")
    ObjectPosition()
    rospy.spin()
