import smach

from typing import List
from shapely.geometry import Polygon

import numpy as np

from lasr_skills import PlayMotion, Detect3DInArea, LookToPoint, Say


class SeatGuest(smach.StateMachine):
    _motions: List[str] = ["look_down_left", "look_down_right", "look_down_centre"]

    class ProcessDetections(smach.State):
        def __init__(self):
            smach.State.__init__(
                self,
                outcomes=["succeeded", "failed"],
                input_keys=[
                    "detections_3d",
                ],
                output_keys=["seat_position"],
            )

        def execute(self, userdata) -> str:
            seat_detections = [
                det for det in userdata.detections_3d if det.name == "chair"
            ]
            person_detections = [
                det for det in userdata.detections_3d if det.name == "person"
            ]

            person_polygons: List[Polygon] = [
                Polygon(np.array(person.xyseg).reshape(-1, 2))
                for person in person_detections
            ]

            print(
                f"There are {len(seat_detections)} seats and {len(person_detections)} people."
            )

            for seat in seat_detections:
                seat_polygon: Polygon = Polygon(np.array(seat.xyseg).reshape(-1, 2))
                seat_is_empty: bool = True
                for person_polygon in person_polygons:
                    if person_polygon.intersects(seat_polygon):
                        seat_is_empty = False
                        print(person_polygon.intersection(seat_polygon))
                        break

                if seat_is_empty:
                    userdata.seat_position = seat.point
                    print(seat.point)
                    return "succeeded"

            return "failed"

    def __init__(
        self,
        seat_area: Polygon,
    ):
        smach.StateMachine.__init__(self, outcomes=["succeeded", "failed"])
        with self:
            # TODO: stop doing this
            self.userdata.people_detections = []
            self.userdata.seat_detections = []

            motion_iterator = smach.Iterator(
                outcomes=["succeeded", "failed"],
                it=self._motions,
                it_label="motion",
                input_keys=["people_detections", "seat_detections"],
                output_keys=["seat_position"],
                exhausted_outcome="failed",
            )

            with motion_iterator:
                container_sm = smach.StateMachine(
                    outcomes=["succeeded", "failed", "continue"],
                    input_keys=["motion", "people_detections", "seat_detections"],
                    output_keys=["seat_position"],
                )

                with container_sm:
                    smach.StateMachine.add(
                        "LOOK",
                        PlayMotion(),
                        transitions={
                            "succeeded": "DETECT",
                            "aborted": "failed",
                            "preempted": "failed",
                        },
                        remapping={"motion_name": "motion"},
                    )
                    smach.StateMachine.add(
                        "DETECT",
                        Detect3DInArea(seat_area, filter=["chair", "person"]),
                        transitions={"succeeded": "CHECK", "failed": "failed"},
                    )
                    smach.StateMachine.add(
                        "CHECK",
                        self.ProcessDetections(),
                        transitions={"succeeded": "succeeded", "failed": "continue"},
                    )

                smach.Iterator.set_contained_state(
                    "CONTAINER_SM", container_sm, loop_outcomes=["continue"]
                )

            smach.StateMachine.add(
                "MOTION_ITERATOR",
                motion_iterator,
                transitions={"succeeded": "LOOK_TO_POINT", "failed": "failed"},
            )
            smach.StateMachine.add(
                "LOOK_TO_POINT",
                LookToPoint(),
                transitions={
                    "succeeded": "SAY_SIT",
                    "aborted": "failed",
                    "preempted": "failed",
                },
                remapping={"point": "seat_position"},
            )
            smach.StateMachine.add(
                "SAY_SIT",
                Say("Please sit in the seat that I am looking at."),
                transitions={
                    "succeeded": "RESET_HEAD",
                    "aborted": "failed",
                    "preempted": "failed",
                },
            )  # TODO: sleep after this.

            smach.StateMachine.add(
                "RESET_HEAD",
                PlayMotion("look_centre"),
                transitions={
                    "succeeded": "succeeded",
                    "aborted": "failed",
                    "preempted": "failed",
                },
            )
