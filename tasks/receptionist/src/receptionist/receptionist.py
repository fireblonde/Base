#!/usr/bin/env python3
import smach
import rospy

from geometry_msgs.msg import Pose, Point, Quaternion
from shapely.geometry import Polygon

from lasr_skills import GoToLocation, WaitForPersonInArea, Say


class Receptionist(smach.StateMachine):

    def __init__(self, wait_pose: Pose, wait_area: Polygon):

        smach.StateMachine.__init__(self, outcomes=["succeeded", "failed"])

        with self:

            smach.StateMachine.add(
                "GO_TO_WAIT_LOCATION",
                GoToLocation(wait_pose),
                transitions={
                    "succeeded": "SAY_WAITING",
                    "failed": "failed",
                },
            )

            smach.StateMachine.add(
                "SAY_WAITING",
                Say("I am waiting for a guest."),
                transitions={
                    "succeeded": "WAIT_FOR_PERSON",
                    "aborted": "failed",
                    "preempted": "failed",
                },
            )

            smach.StateMachine.add(
                "WAIT_FOR_PERSON",
                WaitForPersonInArea(wait_area),
                transitions={
                    "succeeded": "succeeded",
                    "failed": "failed",
                },
            )


if __name__ == "__main__":
    rospy.init_node("receptionist")

    wait_pose_param: Pose = rospy.get_param("/receptionist/wait_pose")
    wait_pose = Pose(
        position=Point(**wait_pose_param["position"]),
        orientation=Quaternion(**wait_pose_param["orientation"]),
    )

    wait_area_param = rospy.get_param("/receptionist/wait_area")
    wait_area = Polygon(wait_area_param)

    sm = Receptionist(wait_pose, wait_area)
    outcome = sm.execute()