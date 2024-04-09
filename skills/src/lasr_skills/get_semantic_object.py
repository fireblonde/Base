import smach
import rospy
from lasr_skills import (
    Listen,
    Say,
    ListenFor,
    ReceiveObject,
    HandoverObject,
    ObjectSemanticInfo,
)

"Get me the object on the left of the mug."


class GetSemanticObject(smach.StateMachine):
    def __init__(self):
        smach.StateMachine.__init__(
            self,
            outcomes=["succeeded", "failed"],
        )

        with self:
            smach.StateMachine.add(
                "LISTEN_FOR_OBJECT",
                ListenFor("left of the mug"),
                transitions={
                    "succeeded": "TALK_TO_OPERATOR",
                    "not_done": "LISTEN_FOR_OBJECT",
                    "aborted": "failed",
                },
            ),
            smach.StateMachine.add(
                "TALK_TO_OPERATOR",
                Say("I heard you want the object on the left of the mug."),
                transitions={
                    "succeeded": "OBJECT_SEMANTIC_INFO",
                    "aborted": "failed",
                },
            ),
            smach.StateMachine.add(
                "OBJECT_SEMANTIC_INFO",
                ObjectSemanticInfo(),
                transitions={
                    "succeeded": "RECEIVE_OBJECT",
                    "failed": "failed",
                },
            ),
            smach.StateMachine.add(
                "RECEIVE_OBJECT",
                ReceiveObject(),
                transitions={
                    "succeeded": "HANDOVER_OBJECT",
                    "failed": "failed",
                },
            ),
            smach.StateMachine.add(
                "HANDOVER_OBJECT",
                HandoverObject(),
                transitions={
                    "succeeded": "succeeded",
                    "failed": "failed",
                },
            )


if __name__ == "__main__":
    rospy.init_node("get_semantic_object")
    sm = GetSemanticObject()
    sm.execute()
    rospy.spin()
