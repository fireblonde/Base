import smach
from smach import UserData
from typing import Union
from lasr_skills import Say
from lasr_vision_clip.srv import VqaRequest, VqaResponse, Vqa
import rospy


class QueryImage(smach.State):
    def __init__(
        self,
        model_device: str = "cuda",
    ):
        smach.State.__init__(
            self,
            outcomes=["succeeded", "failed"],
            input_keys=["question", "answers"],
            output_keys=["answer", "similarity_score"],
        )
        self._service_proxy = rospy.ServiceProxy("/clip_vqa/query_service", Vqa)

    def execute(self, userdata: UserData):
        answers = userdata.answers
        request = VqaRequest()
        request.possible_answers = answers
        response = self._service_proxy(request)
        userdata.answer = response.answer
        userdata.similarity_score = response.similarity
        return "succeeded"
