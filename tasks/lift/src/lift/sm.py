#!/usr/bin/env python3
import smach
# from lift.phases import Phase2CP
from lift.phases import Phase1, Phase2, Phase3, Phase2CP
from lift.default import Default

class Lift(smach.StateMachine):
    def __init__(self):
        smach.StateMachine.__init__(self, outcomes=['success'])
        self.default = Default()

        if self.default.demo_choosing_position:
            print("DEMO CHOOSING POSITION")
            with self:
                smach.StateMachine.add('PHASE2CP', Phase2CP(self.default), transitions={'success' : 'success'})
        else:
            with self:
                smach.StateMachine.add('PHASE1', Phase1(self.default), transitions={'success' : 'PHASE2'})
                smach.StateMachine.add('PHASE2', Phase2(self.default), transitions={'success' : 'PHASE3'})
                smach.StateMachine.add('PHASE3', Phase3(self.default), transitions={'success' : 'success'})