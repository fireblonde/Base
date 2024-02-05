import smach
from .states import GoToLift, CheckOpenDoor, NavigateInLift, WaitForPeople
class Phase2CP(smach.StateMachine):
    def __init__(self, default):
        smach.StateMachine.__init__(self, outcomes=['success'])

        with self:
            smach.StateMachine.add('GO_TO_LIFT', GoToLift(default), transitions={'success': 'CHECK_OPEN_DOOR'})
            smach.StateMachine.add('CHECK_OPEN_DOOR', CheckOpenDoor(default), transitions={'success': 'WAIT_FOR_PEOPLE', 'failed': 'CHECK_OPEN_DOOR'})
            smach.StateMachine.add('WAIT_FOR_PEOPLE', WaitForPeople(default), transitions={'success': 'NAVIGATE_IN_LIFT', 'failed': 'WAIT_FOR_PEOPLE'})
            smach.StateMachine.add('NAVIGATE_IN_LIFT', NavigateInLift(default), transitions={'success': 'success'})


