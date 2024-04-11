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
        pass


class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        L = ray.origin - self.center
        a = glm.dot(ray.direction, ray.direction)
        b = 2 * glm.dot(ray.direction, L)
        c = glm.dot(L, L) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return False
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)
            t = min(filter(lambda x: x > 0, [t1, t2]), default=None)
            if t is not None:
                intersect.time = t
                intersect.point = ray.origin + t * ray.direction
                intersect.normal = glm.normalize(intersect.point - self.center)
                intersect.material = self.materials[0]
                return True
            return False

class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float, center_end: glm.vec3 = None, time_start: float = 0.0, time_end: float = 1.0):
        super().__init__(name, gtype, materials)
        self.center_start = center
        self.center_end = center_end if center_end is not None else center
        self.radius = radius
        self.time_start = time_start
        self.time_end = time_end

    def center_at_time(self, time):
        # If the sphere is stationary, center_start and center_end will be the same, resulting in no movement.
        return self.center_start + (self.center_end - self.center_start) * ((time - self.time_start) / (self.time_end - self.time_start))
    
    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        if self.gtype == 'moving_sphere':
            moving_center = self.center_at_time(ray.time)
            L = ray.origin - moving_center
        else:
            L = ray.origin - self.center_start
        a = glm.dot(ray.direction, ray.direction)
        b = 2 * glm.dot(ray.direction, L)
        c = glm.dot(L, L) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return False
        else:
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)
            t = min(filter(lambda x: x > 0, [t1, t2]), default=None)
            if t is not None:
                intersect.time = t
                intersect.point = ray.origin + t * ray.direction
                if self.gtype == 'moving_sphere':
                    intersect.normal = glm.normalize(intersect.point - moving_center)
                else:
                    intersect.normal = glm.normalize(intersect.point - self.center_start)
                intersect.material = self.materials[0]
                return True
            return False

class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # Calculate the intersection using the plane equation
        denom = glm.dot(self.normal, ray.direction)
        if abs(denom) > 1e-6:
            t = glm.dot(self.point - ray.origin, self.normal) / denom
            if t >= 0:
                intersect.point = ray.origin + t * ray.direction
                intersect.normal = self.normal
                intersect.time = t

                if len(self.materials) == 2:
                    checker_x = int(math.floor(intersect.point.x))
                    checker_z = int(math.floor(intersect.point.z))
                    if (checker_x + checker_z) % 2 == 0:
                        intersect.material = self.materials[0]
                    else:
                        intersect.material = self.materials[1]
                else:
                    intersect.material = self.materials[0]
                return True
        return False


class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, dimension: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        halfside = dimension / 2
        self.minpos = center - halfside
        self.maxpos = center + halfside

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        t_min = (self.minpos - ray.origin) / ray.direction
        t_max = (self.maxpos - ray.origin) / ray.direction
        t1 = glm.min(t_min, t_max)
        t2 = glm.max(t_min, t_max)
        t_near = max(t1.x, t1.y, t1.z)
        t_far = min(t2.x, t2.y, t2.z)
        if t_near > t_far or t_far < epsilon:
            return False
        intersect.point = ray.origin + t_near * ray.direction
        intersect.time = t_near
        # Determine face hit from multiple AABB Faces
        normal = glm.vec3(0, 0, 0)
        for i in range(3):
            if t_near == t1[i]:
                normal[i] = 1 if t_min[i] > t_max[i] else -1
        intersect.normal = normal
        intersect.material = self.materials[0]
        return True


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
        hit = False
        for face in self.faces:
            v0, v1, v2 = self.verts[face[0]], self.verts[face[1]], self.verts[face[2]]
            normal = glm.normalize(glm.cross(v1 - v0, v2 - v0))
            # Ray-plane intersection
            denom = glm.dot(normal, ray.direction)
            if abs(denom) > 1e-6:
                t = glm.dot(v0 - ray.origin, normal) / denom
                if t < 0:
                    continue
                # Compute the intersection point
                P = ray.origin + t * ray.direction
                # Inside-out test
                if glm.dot(normal, glm.cross(v1 - v0, P - v0)) < 0 or glm.dot(normal, glm.cross(v2 - v1, P - v1)) < 0 or glm.dot(normal, glm.cross(v0 - v2, P - v2)) < 0:
                    continue
                if not hit or t < intersect.time:
                    hit = True
                    intersect.time = t
                    intersect.point = P
                    intersect.normal = normal if denom < 0 else -normal
                    intersect.material = self.materials[0]
        return hit


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
        # Transform the ray into the local space of the hierarchy
        transformed_ray = hc.Ray(
            glm.vec3(self.Minv * glm.vec4(ray.origin, 1.0)),
            glm.normalize(glm.vec3(self.Minv * glm.vec4(ray.direction, 0.0)))
        )
        hit = False
        closest_time = float('inf')
        temp_intersect = hc.Intersection.default()  # Temporary intersection to store child intersections

        for child in self.children:
            if child.intersect(transformed_ray, temp_intersect):
                # Transform the intersection point back to world space to calculate correct time
                world_space_point = glm.vec3(self.M * glm.vec4(temp_intersect.point, 1.0))
                world_space_time = glm.length(world_space_point - ray.origin)

                # Check if this intersection is closer than previous ones
                if world_space_time < closest_time:
                    closest_time = world_space_time
                    intersect.time = world_space_time
                    intersect.point = world_space_point
                    intersect.material = temp_intersect.material
                    # Transform the normal back to world space
                    transformed_normal = glm.vec3(glm.transpose(self.Minv) * glm.vec4(temp_intersect.normal, 0.0))
                    intersect.normal = glm.normalize(transformed_normal)
                    hit = True

        return hit

class Quadric(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, coefficients: dict):
        super().__init__(name, gtype, materials)
        self.point = point
        self.coefficients = coefficients  # A dictionary of the coefficients A, B, C, D, E, F, G, H, I, J

    @staticmethod
    def from_json(name: str, g_type: str, materials: list[hc.Material], point: glm.vec3, geometry: dict):
        coefficients = {k: geometry["coefficients"][k] for k in "ABCDEFGHIJ"}
        return Quadric(name, g_type, materials, point, coefficients)
    
    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        # Adjust ray origin by the inverse of the quadric's position
        adjusted_origin = ray.origin - self.point

        # Use the adjusted ray origin for intersection calculations
        A, B, C, D, E, F, G, H, I, J = (self.coefficients[k] for k in "ABCDEFGHIJ")
        xd, yd, zd = ray.direction.x, ray.direction.y, ray.direction.z
        xo, yo, zo = adjusted_origin.x, adjusted_origin.y, adjusted_origin.z

        Aq = A*xd**2 + B*yd**2 + C*zd**2 + D*xd*yd + E*xd*zd + F*yd*zd
        Bq = 2*A*xo*xd + 2*B*yo*yd + 2*C*zo*zd + D*(xo*yd + yo*xd) + E*(xo*zd + zo*xd) + F*(yo*zd + yd*zo) + G*xd + H*yd + I*zd
        Cq = A*xo**2 + B*yo**2 + C*zo**2 + D*xo*yo + E*xo*zo + F*yo*zo + G*xo + H*yo + I*zo + J

        discriminant = Bq**2 - 4*Aq*Cq
        if discriminant < 0:
            return False  # No real roots, no intersection

        sqrt_discriminant = math.sqrt(discriminant)
        t0 = (-Bq - sqrt_discriminant) / (2 * Aq)
        t1 = (-Bq + sqrt_discriminant) / (2 * Aq)

        # Find the closest positive t value
        t = None
        if t0 > epsilon and (t1 <= epsilon or t0 < t1):
            t = t0
        elif t1 > epsilon:
            t = t1

        if t is not None:
            intersect.time = t
            intersect.point = ray.origin + t * ray.direction
            # Compute the normal at the intersection point
            local_intersect_point = intersect.point - self.point
            intersect.normal = glm.vec3(
                2*A*local_intersect_point.x + D*local_intersect_point.y + E*local_intersect_point.z + G,
                2*B*local_intersect_point.y + D*local_intersect_point.x + F*local_intersect_point.z + H,
                2*C*local_intersect_point.z + E*local_intersect_point.x + F*local_intersect_point.y + I
            )
            intersect.normal = glm.normalize(intersect.normal)
            # Adjust the normal direction
            if glm.dot(ray.direction, intersect.normal) > 0:
                intersect.normal = -intersect.normal
            intersect.material = self.materials[0]
            return True

        return False