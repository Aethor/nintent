from enum import Enum, auto


class Intent(Enum):
    COMBINE = auto()
    GET_EVENT_ATTENDEE_AMOUNT = auto()
    GET_EVENT = auto()
    NEGATION = auto()
    GET_LOCATION_SCHOOL = auto()
    GET_DIRECTIONS = auto()
    GET_ESTIMATED_DEPARTURE = auto()
    GET_LOCATION_HOME = auto()
    GET_ESTIMATED_DURATION = auto()
    GET_INFO_TRAFFIC = auto()
    UNSUPPORTED_EVENT = auto()
    GET_ESTIMATED_ARRIVAL = auto()
    GET_INFO_ROUTE = auto()
    GET_LOCATION_WORK = auto()
    UNSUPPORTED_NAVIGATION = auto()
    GET_EVENT_ATTENDEE = auto()
    UNSUPPORTED = auto()
    GET_LOCATION_HOMETOWN = auto()
    GET_DISTANCE = auto()
    GET_EVENT_ORGANIZER = auto()
    UPDATE_DIRECTIONS = auto()
    UNINTELLIGIBLE = auto()
    GET_LOCATION = auto()
    GET_INFO_ROAD_CONDITION = auto()
    GET_CONTACT = auto()


class Slot(Enum):
    WAYPOINT_AVOID = auto()
    COMBINE = auto()
    ROAD_CONDITION_AVOID = auto()
    CONTACT = auto()
    PATH_AVOID = auto()
    TYPE_RELATION = auto()
    LOCATION_CURRENT = auto()
    AMOUNT = auto()
    NAME_EVENT = auto()
    DATE_TIME = auto()
    DATE_TIME_ARRIVAL = auto()
    DATE_TIME_DEPARTURE = auto()
    SEARCH_RADIUS = auto()
    OBSTRUCTION_AVOID = auto()
    POINT_ON_MAP = auto()
    GROUP = auto()
    SOURCE = auto()
    DESTINATION = auto()
    CATEGORY_LOCATION = auto()
    METHOD_TRAVEL = auto()
    ORDINAL = auto()
    OBSTRUCTION = auto()
    CONTACT_RELATED = auto()
    UNIT_DISTANCE = auto()
    ATTRIBUTE_EVENT = auto()
    WAYPOINT = auto()
    LOCATION_USER = auto()
    LOCATION = auto()
    ATTENDEE_EVENT = auto()
    ORGANIZER_EVENT = auto()
    ROAD_CONDITION = auto()
    PATH = auto()
    LOCATION_MODIFIER = auto()
    LOCATION_WORK = auto()
    WAYPOINT_ADDED = auto()
    CATEGORY_EVENT = auto()
