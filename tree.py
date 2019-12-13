from __future__ import annotations
from typing import Union, Tuple, Optional, List


class Intent:
    intent_types = {
        1: "NOT_INTENT",
        2: "COMBINE",
        3: "GET_EVENT_ATTENDEE_AMOUNT",
        4: "GET_EVENT",
        5: "NEGATION",
        6: "GET_LOCATION_SCHOOL",
        7: "GET_DIRECTIONS",
        8: "GET_ESTIMATED_DEPARTURE",
        9: "GET_LOCATION_HOME",
        10: "GET_ESTIMATED_DURATION",
        11: "GET_INFO_TRAFFIC",
        12: "UNSUPPORTED_EVENT",
        13: "GET_ESTIMATED_ARRIVAL",
        14: "GET_INFO_ROUTE",
        15: "GET_LOCATION_WORK",
        16: "UNSUPPORTED_NAVIGATION",
        17: "GET_EVENT_ATTENDEE",
        18: "UNSUPPORTED",
        19: "GET_LOCATION_HOMETOWN",
        20: "GET_DISTANCE",
        21: "GET_EVENT_ORGANIZER",
        22: "UPDATE_DIRECTIONS",
        23: "UNINTELLIGIBLE",
        24: "GET_LOCATION",
        25: "GET_INFO_ROAD_CONDITION",
        26: "GET_CONTACT",
    }

    def __init__(self, intent_type: Union[str, int]):
        if isinstance(intent_type, int):
            self.type = Intent.intent_types[intent_type]
        else:
            try:
                self.type = {v: k for k, v in Intent.intent_types.items()}[intent_type]
            except KeyError:
                raise Exception(f"Invalid intent type {intent_type}")

    def __str__(self):
        return Intent.intent_types[self.type]


class Slot:
    slot_types = {
        1: "NOT_SLOT",
        2: "WAYPOINT_AVOID",
        3: "COMBINE",
        4: "ROAD_CONDITION_AVOID",
        5: "CONTACT",
        6: "PATH_AVOID",
        7: "TYPE_RELATION",
        8: "LOCATION_CURRENT",
        9: "AMOUNT",
        10: "NAME_EVENT",
        11: "DATE_TIME",
        12: "DATE_TIME_ARRIVAL",
        13: "DATE_TIME_DEPARTURE",
        14: "SEARCH_RADIUS",
        15: "OBSTRUCTION_AVOID",
        16: "POINT_ON_MAP",
        17: "GROUP",
        18: "SOURCE",
        19: "DESTINATION",
        20: "CATEGORY_LOCATION",
        21: "METHOD_TRAVEL",
        22: "ORDINAL",
        23: "OBSTRUCTION",
        24: "CONTACT_RELATED",
        25: "UNIT_DISTANCE",
        26: "ATTRIBUTE_EVENT",
        27: "WAYPOINT",
        28: "LOCATION_USER",
        29: "LOCATION",
        30: "ATTENDEE_EVENT",
        31: "ORGANIZER_EVENT",
        32: "ROAD_CONDITION",
        33: "PATH",
        34: "LOCATION_MODIFIER",
        35: "LOCATION_WORK",
        36: "WAYPOINT_ADDED",
        37: "CATEGORY_EVENT",
    }

    def __init__(self, slot_type: Union[str, int]):
        if isinstance(slot_type, int):
            self.type = Slot.slot_types[slot_type]
        else:
            try:
                self.type = {v: k for k, v in Slot.slot_types.items()}[slot_type]
            except KeyError:
                raise Exception(f"Invalid slot type {slot_type}")

    def __str__(self):
        return Slot.slot_types[self.type]


class IntentTree:
    def __init__(self, tokens: str, node_type: Tuple[Union[Intent, Slot]]):
        self.tokens = tokens
        self.node_type = node_type
        self.children = []

    def add_child(self, child: IntentTree):
        self.children.append(child)

    @classmethod
    def from_str(self, sent: str) -> IntentTree:
        stack: List[IntentTree] = []

        for token in sent.split():

            if token.startswith("["):
                label, typ = token[1:].split(":")
                if label == "IN":
                    stack.append(IntentTree("", (Intent(typ))))
                elif label == "SL":
                    stack.append(IntentTree("", (Slot(typ))))
                else:
                    raise Exception(f"Unknown node type : {label}")
                if len(stack) >= 2:
                    stack[-2].add_child(stack[-1])

            elif token.startswith("]"):
                last_popped = stack.pop()

            else:
                for node in stack:
                    node.tokens += f" {token}"

        return last_popped

    def __str__(self, level: int = 0):
        offset = "  " * level
        string = offset + "+ {} : {}\n".format(str(self.node_type), self.tokens)
        for child in self.children:
            string += child.__str__(level=level + 1)
        return string
