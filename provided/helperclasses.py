# Jerry Xia 260917329

import glm

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry


class Ray:
    def __init__(self, o: glm.vec3, d: glm.vec3, time=0.0):
        self.origin = o
        self.direction = d
        self.time = time

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t


class Material:
    def __init__(self, name: str, specular: glm.vec3, diffuse: glm.vec3, hardness: float, ID: int, is_mirror: bool, is_transparent: bool, ior: float):
        self.name = name
        self.specular = specular
        self.diffuse = diffuse
        self.hardness = hardness
        self.ID = ID
        self.is_mirror = is_mirror
        self.is_transparent = is_transparent
        self.ior = ior

    @staticmethod
    def default():
        name = "default"
        specular = diffuse = glm.vec3(0, 0, 0)
        hardness = ID = -1
        is_mirror = False
        is_transparent = False
        ior = 1.0
        return Material(name, specular, diffuse, hardness, ID, is_mirror, is_transparent, ior)


class Light:
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, power: float):
        self.type = ltype
        self.name = name
        self.colour = colour
        self.vector = vector
        self.power = power


class Intersection:

    def __init__(self, time: float, normal: glm.vec3, position: glm.vec3, material: Material):
        self.time = time
        self.normal = normal
        self.position = position
        self.mat = material

    @staticmethod
    def default():
        time = float("inf")
        normal = glm.vec3(0, 0, 0)
        position = glm.vec3(0, 0, 0)
        mat = Material.default()
        return Intersection(time, normal, position, mat)
