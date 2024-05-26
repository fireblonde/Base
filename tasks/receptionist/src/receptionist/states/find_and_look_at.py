"""
State machine for the nicole task. It finds a person by given name and then looks at them.
"""
import rospy
import smach_ros
import smach
from lasr_skills import LookAtPerson, LookToPoint
from typing import List, Union
from geometry_msgs.msg import Point, PointStamped
from lasr_vision_msgs.srv import Recognise, RecogniseRequest
from lasr_skills.vision import GetPointCloud
from cv2_pcl import pcl_to_img_msg
import actionlib
from geometry_msgs.msg import Point
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from smach import CBState
import rosservice

PUBLIC_CONTAINER = False

try:
    from pal_startup_msgs.srv import (
        StartupStart,
        StartupStop,
        StartupStartRequest,
        StartupStopRequest,
    )
except ModuleNotFoundError:
    PUBLIC_CONTAINER = True


def send_head_goal(_point, look_at_pub):
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ["head_1_joint", "head_2_joint"]
    point = JointTrajectoryPoint()
    point.positions = _point
    point.time_from_start = rospy.Duration(1)
    goal.trajectory.points.append(point)
    look_at_pub.send_goal(goal)


class FindAndLookAt(smach.StateMachine):
    class GetLookPoint(smach.State):
        def __init__(self, look_positions: List[List[float]]):
            smach.State.__init__(
                self,
                outcomes=["succeeded", "failed"],
                input_keys=[],
                output_keys=["look_positions"],
            )
            self.look_positions = look_positions

        def execute(self, userdata):
            userdata.look_positions = self.look_positions
            return "succeeded"

    class GetPoint(smach.State):
        def __init__(self):
            smach.State.__init__(
                self,
                outcomes=["succeeded", "failed"],
                input_keys=["point_index", "look_positions"],
                output_keys=["pointstamped"],
            )
            self.look_at_pub = actionlib.SimpleActionClient(
                "/head_controller/follow_joint_trajectory", FollowJointTrajectoryAction
            )

        def execute(self, userdata):
            rospy.sleep(3.0)
            _point = userdata.look_positions[userdata.point_index]
            print(f"Looking at {_point}")
            userdata.pointstamped = PointStamped(
                point=Point(x=_point[0], y=_point[1], z=1.0)
            )
            send_head_goal(_point, self.look_at_pub)
            rospy.sleep(3.0)

            return "succeeded"

    def check_name(self, ud):
        rospy.logwarn(
            f"Checking name {ud.guest_name} in detections {ud.deepface_detection}"
        )
        if len(ud.deepface_detection) == 0:
            return "no_detection"
        for detection in ud.deepface_detection:
            if detection.name == ud.guest_name and detection.confidence > ud.confidence:
                return "succeeded"
        return "failed"

    def __init__(
        self,
        look_positions: Union[List[List[float]], None] = None,
    ):
        smach.StateMachine.__init__(
            self,
            outcomes=["succeeded", "failed"],
            input_keys=["guest_name", "dataset", "confidence"],
            output_keys=[],
        )

        if look_positions is None:
            all_look_positions: List[List[float]] = []
            look_positions = [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
            ]

        all_look_positions = look_positions
        IS_SIMULATION = (
            "/pal_startup_control/start" not in rosservice.get_service_list()
        )

        with self:
            smach.StateMachine.add(
                "GET_LOOK_POINT",
                self.GetLookPoint(all_look_positions),
                transitions={"succeeded": "LOOK_ITERATOR", "failed": "failed"},
            )
            look_iterator = smach.Iterator(
                outcomes=["succeeded", "failed"],
                it=lambda: range(len(all_look_positions)),
                it_label="point_index",
                input_keys=["look_positions", "dataset", "confidence", "guest_name"],
                output_keys=[],
                exhausted_outcome="failed",
            )
            with look_iterator:
                container_sm = smach.StateMachine(
                    outcomes=["succeeded", "failed", "continue"],
                    input_keys=[
                        "point_index",
                        "look_positions",
                        "dataset",
                        "confidence",
                        "guest_name",
                    ],
                    output_keys=[],
                )

                with container_sm:
                    if not IS_SIMULATION:
                        if PUBLIC_CONTAINER:
                            rospy.logwarn(
                                "You are using a public container. The head manager will not be stopped during navigation."
                            )
                        else:
                            smach.StateMachine.add(
                                "DISABLE_HEAD_MANAGER",
                                smach_ros.ServiceState(
                                    "/pal_startup_control/stop",
                                    StartupStop,
                                    request=StartupStopRequest("head_manager"),
                                ),
                                transitions={
                                    "succeeded": "GET_POINT",
                                    "aborted": "failed",
                                    "preempted": "failed",
                                },
                            )
                    smach.StateMachine.add(
                        "GET_POINT",
                        self.GetPoint(),
                        transitions={"succeeded": "GET_IMAGE", "failed": "failed"},
                        remapping={"pointstamped": "pointstamped"},
                    )
                    # smach.StateMachine.add(
                    #     "LOOK_TO_POINT",
                    #     LookToPoint(),
                    #     transitions={
                    #         "succeeded": "GET_IMAGE",
                    #         "aborted": "failed",
                    #         "preempted": "failed",
                    #     },
                    # )
                    smach.StateMachine.add(
                        "GET_IMAGE",
                        GetPointCloud("/xtion/depth_registered/points"),
                        transitions={
                            "succeeded": "RECOGNISE",
                        },
                        remapping={"pcl_msg": "pcl_msg"},
                    )
                    smach.StateMachine.add(
                        "RECOGNISE",
                        smach_ros.ServiceState(
                            "/recognise",
                            Recognise,
                            input_keys=["pcl_msg", "dataset", "confidence"],
                            request_cb=lambda ud, _: RecogniseRequest(
                                image_raw=pcl_to_img_msg(ud.pcl_msg),
                                dataset=ud.dataset,
                                confidence=ud.confidence,
                            ),
                            response_slots=["detections"],
                            output_keys=["detections"],
                        ),
                        transitions={
                            "succeeded": "CHECK_NAME",
                            "aborted": "failed",
                            "preempted": "failed",
                        },
                        remapping={
                            "pcl_msg": "pcl_msg",
                            "detections": "deepface_detection",
                        },
                    )
                    smach.StateMachine.add(
                        "CHECK_NAME",
                        CBState(
                            self.check_name,
                            outcomes=["succeeded", "failed", "no_detection"],
                            input_keys=[
                                "deepface_detection",
                                "guest_name",
                                "confidence",
                            ],
                        ),
                        transitions={
                            "succeeded": "LOOK_AT_PERSON",
                            "failed": "GET_IMAGE",
                            "no_detection": "continue",
                        },
                    )
                    smach.StateMachine.add(
                        "LOOK_AT_PERSON",
                        LookAtPerson(filter=True),
                        transitions={
                            "succeeded": "succeeded",
                            "no_detection": "continue",
                            "failed": "failed",
                        },
                    )
                look_iterator.set_contained_state(
                    "CONTAINER_STATE", container_sm, loop_outcomes=["continue"]
                )
            smach.StateMachine.add(
                "LOOK_ITERATOR",
                look_iterator,
                transitions={"succeeded": "succeeded", "failed": "failed"},
            )
