#!/usr/bin/env python3
import rospy
from receptionist.state_machine import Receptionist

from geometry_msgs.msg import Pose, Point, Quaternion
from shapely.geometry import Polygon

if __name__ == "__main__":
    rospy.init_node("receptionist_robocup")
    wait_pose_param = rospy.get_param("/wait_pose")
    wait_pose = Pose(
        position=Point(**wait_pose_param["position"]),
        orientation=Quaternion(**wait_pose_param["orientation"]),
    )

    wait_area_param = rospy.get_param("/wait_area")
    wait_area = Polygon(wait_area_param)

    seat_pose_param = rospy.get_param("/seat_pose")
    seat_pose = Pose(
        position=Point(**seat_pose_param["position"]),
        orientation=Quaternion(**seat_pose_param["orientation"]),
    )

    seat_area_param = rospy.get_param("/seat_area")
    seat_area = Polygon(seat_area_param)

    receptionist = Receptionist(
        wait_pose,
        wait_area,
        seat_pose,
        seat_area,
        {
            "name": "nicole",
            "drink": "beer",
            "attributes": {
                "hair_colour": "strawberry blonde",
                "glasses": False,
                "hat": True,
                "height": "tall",
            },
            "dataset": "receptionist",
            "confidence": 0.5,
        },
    )
    outcome = receptionist.execute()
    rospy.loginfo(f"Receptionist finished with outcome: {outcome}")
    rospy.spin()
