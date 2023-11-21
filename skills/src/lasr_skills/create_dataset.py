#!/usr/bin/env python3

import smach
from lasr_vision_face_recognition.dataset_collector import DatasetCollector
import rospy

class CreateDataset(smach.State):

    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'failed'], input_keys=['name'])

        if rospy.get_published_topics(namespace='/pal_head_manager'):
            topic = '/xtion/rgb/image_raw'
        else:
            topic = '/usb_cam/image_raw'

        self.dc = DatasetCollector(topic)


    def execute(self, userdata):
        dataset_path, _  = self.dc.collect(userdata.name)
        if dataset_path is None:
            return 'failed'
        return 'succeeded'


if __name__ == "__main__":
    rospy.init_node("create_dataset")
    sm = smach.StateMachine(outcomes=['succeeded', 'failed'])
    sm.userdata.name = "zoe"

    with sm:
        smach.StateMachine.add('CREATE_DATASET', CreateDataset(), transitions={'succeeded' : 'succeeded', 'failed' : 'failed'})

    sm.execute()
