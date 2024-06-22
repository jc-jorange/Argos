from enum import Enum, unique

from .multi_camera_match.center_ray_intersect import CenterRayIntersectMatchor


@unique
class E_MatchorFactory(Enum):
    CenterRayIntersect = 1


factory_matchor = {
    E_MatchorFactory.CenterRayIntersect.name: CenterRayIntersectMatchor,
}

