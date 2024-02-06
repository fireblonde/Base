#!/usr/bin/env python3


import rospy
from actionlib import SimpleActionClient


from qualification.msg import (
    GetObjectPositionAction,
    GetObjectPositionGoal,
)


rospy.init_node("object_loc")

print("Waiting for action servers...")
print("object_position")
object_position = SimpleActionClient("object_position", GetObjectPositionAction)
object_position.wait_for_server()

print("Action servers ready")

goal = GetObjectPositionGoal(["cup", "bottle"])

object_position.send_goal_and_wait(goal)
object_position_result = object_position.get_result()

