#!/usr/bin/env python3
import rospy
import smach

from lasr_skills import DetectObjects
from lasr_vision_face_recognition.srv import Identify

class IdentifyPeople(smach.StateMachine):

    class NamePeople(smach.State):

        def __init__(self):
            smach.State.__init__(self, outcomes=['done', 'not_done'], input_keys=['img_msg', 'detections'], output_keys=['identified_people'])
            self.identify_people = rospy.ServiceProxy('/lasr_vision/identify_people', Identify)

        def execute(self, userdata):
            try:
                result = self.identify_people(userdata.img_msg, userdata.detections.detected_objects)
                if not result.success:
                    print("No people identified")
                    rospy.sleep(10)
                    return 'not_done'
                userdata.identified_people = result.identified_people
                if len(userdata.detections.detected_objects) > userdata.identified_people:
                    print("Not all people identified")
                    names = [det.name for det in userdata.identified_people]
                    print(f'I only know {names}')
                return 'done'
            except rospy.ServiceException as e:
                rospy.sleep(10)
                rospy.logwarn(f"Unable to perform inference. ({str(e)})")
                return 'not_done'

    def __init__(self):
        smach.StateMachine.__init__(self, outcomes=['succeeded', 'failed'], output_keys=['identified_people'])

        self.userdata.image_topic = "/xtion/rgb/image_raw"
        # self.userdata.image_topic = '/usb_cam/image_raw'
        self.userdata.filter = ["person"]


        with self:
            smach.StateMachine.add('DETECT_OBJECTS', DetectObjects(), transitions={'succeeded': 'NAME_PEOPLE', 'failed': 'DETECT_OBJECTS'})
            smach.StateMachine.add('NAME_PEOPLE', self.NamePeople(), transitions={'done': 'succeeded', 'not_done': 'DETECT_OBJECTS'})

if __name__ == '__main__':
    rospy.init_node('identify_people')
    sm = IdentifyPeople()
    sm.execute()
