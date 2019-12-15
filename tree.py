from __future__ import annotations
from typing import Union, Tuple, Optional, List


class Intent:
    intent_types = {
        0: "NOT_INTENT",
        1: "COMBINE",
        2: "GET_EVENT_ATTENDEE_AMOUNT",
        3: "GET_EVENT",
        4: "NEGATION",
        5: "GET_LOCATION_SCHOOL",
        6: "GET_DIRECTIONS",
        7: "GET_ESTIMATED_DEPARTURE",
        8: "GET_LOCATION_HOME",
        9: "GET_ESTIMATED_DURATION",
        10: "GET_INFO_TRAFFIC",
        11: "UNSUPPORTED_EVENT",
        12: "GET_ESTIMATED_ARRIVAL",
        13: "GET_INFO_ROUTE",
        14: "GET_LOCATION_WORK",
        15: "UNSUPPORTED_NAVIGATION",
        16: "GET_EVENT_ATTENDEE",
        17: "UNSUPPORTED",
        18: "GET_LOCATION_HOMETOWN",
        19: "GET_DISTANCE",
        20: "GET_EVENT_ORGANIZER",
        21: "UPDATE_DIRECTIONS",
        22: "UNINTELLIGIBLE",
        23: "GET_LOCATION",
        24: "GET_INFO_ROAD_CONDITION",
        25: "GET_CONTACT",
    }

    def __init__(self, intent_type: Union[str, int]):
        if isinstance(intent_type, int):
            self.type = Intent.itos(intent_type)
        else:
            if not intent_type in {v: k for k, v in Intent.intent_types.items()}:
                raise KeyError(f"Invalid intent type {intent_type}")
            self.type = intent_type

    @classmethod
    def itos(cls, i: int):
        return Intent.intent_types[i]

    @classmethod
    def stoi(cls, s: str):
        return {v: k for k, v in Intent.intent_types.items()}[s]

    def __eq__(self, other: Intent):
        if not isinstance(other, Intent):
            return False
        return other.type == self.type

    def __str__(self):
        return "INTENT : " + Intent.intent_types[self.type]


class Slot:
    slot_types = {
        0: "NOT_SLOT",
        1: "WAYPOINT_AVOID",
        2: "COMBINE",
        3: "ROAD_CONDITION_AVOID",
        4: "CONTACT",
        5: "PATH_AVOID",
        6: "TYPE_RELATION",
        7: "LOCATION_CURRENT",
        8: "AMOUNT",
        9: "NAME_EVENT",
        10: "DATE_TIME",
        11: "DATE_TIME_ARRIVAL",
        12: "DATE_TIME_DEPARTURE",
        13: "SEARCH_RADIUS",
        14: "OBSTRUCTION_AVOID",
        15: "POINT_ON_MAP",
        16: "GROUP",
        17: "SOURCE",
        18: "DESTINATION",
        19: "CATEGORY_LOCATION",
        20: "METHOD_TRAVEL",
        21: "ORDINAL",
        22: "OBSTRUCTION",
        23: "CONTACT_RELATED",
        24: "UNIT_DISTANCE",
        25: "ATTRIBUTE_EVENT",
        26: "WAYPOINT",
        27: "LOCATION_USER",
        28: "LOCATION",
        29: "ATTENDEE_EVENT",
        30: "ORGANIZER_EVENT",
        31: "ROAD_CONDITION",
        32: "PATH",
        33: "LOCATION_MODIFIER",
        34: "LOCATION_WORK",
        35: "WAYPOINT_ADDED",
        36: "CATEGORY_EVENT",
    }

    def __init__(self, slot_type: Union[str, int]):
        if isinstance(slot_type, int):
            self.type = Slot.itos(slot_type)
        else:
            if not slot_type in {v: k for k, v in Slot.slot_types.items()}:
                raise KeyError(f"Invalid slot type {slot_type}")
            self.type = slot_type

    @classmethod
    def itos(cls, i: int):
        return Slot.slot_types[i]

    @classmethod
    def stoi(cls, s: str):
        return {v: k for k, v in Slot.slot_types.items()}[s]

    def __eq__(self, other: Slot):
        if not isinstance(other, Slot):
            return False
        return other.type == self.type

    def __str__(self):
        return "SLOT : " + Slot.slot_types[self.type]


class IntentTree:
    def __init__(self, tokens: str, node_type: Optional[Union[Intent, Slot]]):
        self.tokens = tokens
        self.node_type = node_type
        self.children = []

    def add_child(self, child: IntentTree):
        self.children.append(child)

    @classmethod
    def from_str(cls, sent: str) -> IntentTree:
        stack: List[IntentTree] = []

        for token in sent.split():

            if token.startswith("["):
                label, typ = token[1:].split(":")
                if label == "IN":
                    stack.append(IntentTree("", Intent(typ)))
                elif label == "SL":
                    stack.append(IntentTree("", Slot(typ)))
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

    def __eq__(self, other: IntentTree):
        if not isinstance(other, IntentTree):
            return False
        if len(self.children) != len(other.children):
            return False
        if len(self.children) == 0 and len(other.children) == 0:
            return self.tokens == other.tokens and self.node_type == other.node_type
        for child, other_child in zip(self.children, other.children):
            if child != other_child:
                return False
        return True

    def __str__(self, level: int = 0):
        offset = "  " * level
        string = offset + "+ {} | {}\n".format(str(self.node_type), self.tokens)
        for child in self.children:
            string += child.__str__(level=level + 1)
        return string
