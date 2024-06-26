import smach

from lasr_skills import Detect
from lasr_skills.vision import GetImage


class WaitForPerson(smach.StateMachine):
    class CheckForPerson(smach.State):
        def __init__(self):
            smach.State.__init__(
                self, outcomes=["done", "not_done"], input_keys=["detections"]
            )

        def execute(self, userdata):
            if len(userdata.detections.detected_objects):
                return "done"
            else:
                return "not_done"

    def __init__(
        self,
        image_topic: str = "/xtion/rgb/image_raw",
    ):
        smach.StateMachine.__init__(
            self,
            outcomes=["succeeded", "failed"],
            output_keys=["detections"],
        )

        with self:
            smach.StateMachine.add(
                "GET_IMAGE",
                GetImage(topic=image_topic),
                transitions={"succeeded": "DETECT_PEOPLE", "failed": "failed"},
            )

            smach.StateMachine.add(
                "DETECT_PEOPLE",
                Detect(filter=["person"]),
                transitions={"succeeded": "CHECK_FOR_PERSON", "failed": "failed"},
            )
            smach.StateMachine.add(
                "CHECK_FOR_PERSON",
                self.CheckForPerson(),
                transitions={"done": "succeeded", "not_done": "DETECT_PEOPLE"},
            )
