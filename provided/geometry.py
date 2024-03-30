import math
import helperclasses as hc
import glm
import igl

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

epsilon = 10 ** (-4)


class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        if self.gtype == 'sphere':
            return Sphere.intersect(self, ray, intersect)
        elif self.gtype == 'plane':
            return Plane.intersect(self, ray, intersect)
        elif self.gtype == 'cube':
            return AABB.intersect(self, ray, intersect)
        elif self.gtype == 'mesh':
            return Mesh.intersect(self, ray, intersect)
        elif self.gtype == 'hierarchy':
            return Hierarchy.intersect(self, ray, intersect)
        else:
            raise NotImplementedError(f"Intersection not implemented for {self.gtype}")


class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # TODO: Create intersect code for Sphere
        p = ray.origin
        d = ray.direction
        b = 2 * glm.dot(p, d)
        c = glm.dot(p, p) - self.radius ** 2
        discriminant = b ** 2 - 4 * c

        if discriminant < 0:
            return False
        else:
            t1 = (-b - math.sqrt(discriminant)) / 2
            t2 = (-b + math.sqrt(discriminant)) / 2
            # Choose the smallest positive t value as the intersection point
            t = min(filter(lambda x: x > 0, [t1, t2]), default=None)
            if t is not None:
                intersect.time = t
                intersect.point = p + t * d
                intersect.normal = glm.normalize(intersect.point - self.center)
                intersect.material = self.materials[0]
                return True
            return False


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Plane


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        halfside = dimension / 2
        self.minpos = center - halfside
        self.maxpos = center + halfside

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Cube


class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        for n in norms:
            self.norms.append(glm.vec3(n[0], n[1], n[2]))

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Mesh


class Hierarchy(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], t: glm.vec3, r: glm.vec3, s: glm.vec3):
        super().__init__(name, gtype, materials)
        self.t = t
        self.M = glm.mat4(1.0)
        self.Minv = glm.mat4(1.0)
        self.make_matrices(t, r, s)
        self.children: list[Geometry] = []

    def make_matrices(self, t: glm.vec3, r: glm.vec3, s: glm.vec3):
        self.M = glm.mat4(1.0)
        self.M = glm.translate(self.M, t)
        self.M = glm.rotate(self.M, glm.radians(r.x), glm.vec3(1, 0, 0))
        self.M = glm.rotate(self.M, glm.radians(r.y), glm.vec3(0, 1, 0))
        self.M = glm.rotate(self.M, glm.radians(r.z), glm.vec3(0, 0, 1))
        self.M = glm.scale(self.M, s)
        self.Minv = glm.inverse(self.M)
        self.t = t

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        pass
        # TODO: Create intersect code for Hierarchy
