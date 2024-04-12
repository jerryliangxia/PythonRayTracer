# Jerry Xia 260917329

import math

import glm
import numpy as np
import random

import geometry as geom
import helperclasses as hc

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

shadow_epsilon = 10**(-6)

class Scene:

    def __init__(self,
                 width: int,
                 height: int,
                 jitter: bool,
                 samples: int,
                 position: glm.vec3,
                 lookat: glm.vec3,
                 up: glm.vec3,
                 fov: float,
                 dof: bool,
                 aperture: float,
                 focus_dist: float,
                 ambient: glm.vec3,
                 lights: list[hc.Light],
                 soft_shadows: bool,
                 light_size: float,
                 materials: list[hc.Material],
                 objects: list[geom.Geometry]
                 ):
        self.width = width  # width of image
        self.height = height  # height of image
        self.aspect = width / height  # aspect ratio
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.position = position  # camera position in 3D
        self.lookat = lookat  # camera look at vector
        self.up = up  # camera up position
        self.fov = fov  # camera field of view
        self.dof = dof  # depth of field
        self.aperture = aperture  # aperture for depth of field
        self.focus_dist = focus_dist  # focus distance for depth of field
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # base lights in the scene
        self.soft_shadows = soft_shadows  # soft shadows
        self.light_size = light_size  # light size
        self.materials = materials  # base materials of objects in the scene
        self.objects = objects  # base objects in the scene
        self.shadow_epsilon = 1e-3
        self.max_depth = 50
        self.offsets = self.generate_offsets(self.light_size)
    
    def generate_offsets(self, light_size):
        offsets = [
            glm.vec3(-light_size, -light_size, -light_size),
            glm.vec3(light_size, -light_size, -light_size),
            glm.vec3(-light_size, light_size, -light_size),
            glm.vec3(light_size, light_size, -light_size),
            glm.vec3(-light_size, -light_size, light_size),
            glm.vec3(light_size, -light_size, light_size),
            glm.vec3(-light_size, light_size, light_size),
            glm.vec3(light_size, light_size, light_size)
        ]
        return offsets

    def random_in_unit_disk(self):
        while True:
            p = np.array([random.uniform(-1, 1), random.uniform(-1, 1), 0])
            if np.dot(p, p) < 1:
                return p
    
    def generate_ray(self, i, j, left, right, top, bottom, u, v, w, d):
        # Calculate the ray direction as before
        if self.jitter:
            # Antialiasing: add random offset to i and j
            random_i = i + random.random()
            random_j = j + random.random()
        else:
            random_i = i
            random_j = j
        u_coord = left + (right - left) * (random_i + 0.5) / self.width
        v_coord = bottom + (top - bottom) * (random_j + 0.5) / self.height
        ray_dir = glm.normalize(u * u_coord + v * v_coord - w * d)
        
        # Calculate the ray origin for defocus blur
        if self.dof:
            rd = self.aperture / 2 * self.random_in_unit_disk()
            offset = u * rd[0] + v * rd[1]
            # Adjust the ray direction for the defocus blur
            focus_point = self.position + self.focus_dist * ray_dir
            ray_origin = self.position + offset
            ray_dir = glm.normalize(focus_point - ray_origin)
        else:
            ray_origin = self.position

        ray_time = random.uniform(0.0, 0.2)
        return hc.Ray(ray_origin, ray_dir, ray_time)
    
    def find_closest_intersection(self, ray):
        closest_t = float('inf')
        closest_intersection = hc.Intersection.default()
        for obj in self.objects:
            intersection = hc.Intersection.default()
            if obj.intersect(ray, intersection) and intersection.time < closest_t:
                closest_t = intersection.time
                closest_intersection = intersection
        return closest_intersection
    
    def calculate_light_contribution(self, intersection, light, light_offset):
        colour = glm.vec3(0, 0, 0)

        # Ambient
        colour += self.ambient * light.colour

        # Diffuse
        L = glm.normalize(light.vector + light_offset - intersection.point)
        N = intersection.normal
        diffuse_intensity = max(glm.dot(N, L), 0)
        colour += intersection.material.diffuse * light.colour * diffuse_intensity

        # Specular
        V = glm.normalize(self.position - intersection.point)
        H = glm.normalize(L + V)
        dot_NH = glm.dot(N, H)
        epsilon = 1e-6

        if dot_NH > epsilon:
            specular_intensity = dot_NH ** intersection.material.hardness
        else:
            specular_intensity = 0.0
        colour += intersection.material.specular * light.colour * specular_intensity

        return colour

    
    def is_in_shadow(self, intersection, light, light_offset):
        offset_position = intersection.point + intersection.normal * self.shadow_epsilon
        to_light = light.vector + light_offset - offset_position
        shadow_ray = hc.Ray(offset_position, to_light)
        for obj in self.objects:
            if obj.intersect(shadow_ray, hc.Intersection.default()):
                return True
        return False
    
    def trace_ray(self, ray, depth=0):
        if depth > self.max_depth:  # Avoid infinite recursion
            return glm.vec3(0, 0, 0)

        intersection = self.find_closest_intersection(ray)
        if intersection.time < float('inf'):
            if intersection.material.is_mirror:
                reflection_direction = self.reflect(ray.direction, intersection.normal)
                reflection_ray = hc.Ray(intersection.point + self.shadow_epsilon * intersection.normal, reflection_direction)
                reflection_color = self.trace_ray(reflection_ray, depth + 1)
                return reflection_color
            elif intersection.material.is_transparent:
                return self.refract(ray, intersection, depth + 1)
            else:
                # Calculate direct lighting
                colour = glm.vec3(0, 0, 0)  # Ambient is only added once
                for light in self.lights:
                    if self.soft_shadows:
                        for _, light_offset in enumerate(self.offsets):
                            if not self.is_in_shadow(intersection, light, light_offset):
                                colour += 1 / len(self.offsets) * self.calculate_light_contribution(intersection, light, light_offset) * light.power
                            else:
                                colour += 1 / len(self.offsets) * self.ambient * light.colour * light.power
                    else:
                        if not self.is_in_shadow(intersection, light, glm.vec3(0, 0, 0)):
                            colour += self.calculate_light_contribution(intersection, light, glm.vec3(0, 0, 0)) * light.power
                        else:
                            colour += self.ambient * light.power * light.colour
                        colour = glm.clamp(colour, 0.0, 1.0)
                return colour
        else:
            return glm.vec3(0, 0, 0)

    def reflect(self, direction, normal):
        return direction - 2 * glm.dot(direction, normal) * normal

    def calc_refracting_ray(self, ray, intersection):
        normal = intersection.normal
        n1, n2, refl, trans = 1.0, intersection.material.ior, 0.0, 0.0
        cosI = glm.dot(glm.normalize(ray.direction), normal)

        if cosI > 0.0:
            n1, n2 = n2, n1
            normal = -normal  # Inverting the normal
            cosI = -cosI
        n = n1 / n2
        sinT2 = n ** 2 * (1.0 - cosI ** 2)
        cosT = math.sqrt(max(0.0, 1.0 - sinT2))  # Ensure non-negative for sqrt

        # Fresnel equations
        rn = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT)
        rt = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT)
        rn **= 2
        rt **= 2

        if n == 1.0 or cosT ** 2 < 0.0:  # Total internal reflection or same medium
            return self.calc_reflecting_ray(ray, intersection)

        dir = n * glm.normalize(ray.direction) + (n * cosI - cosT) * normal
        dir = glm.normalize(dir)  # Ensure direction is normalized
        return hc.Ray(intersection.normal + dir * self.shadow_epsilon, dir)

    def refract(self, ray, intersection, depth):
        # Calculate the refracted ray direction (simplified, actual calculation is more complex)
        refracted_direction = self.calc_refracting_ray(ray, intersection)

        # Offset the intersection point slightly to avoid self-intersection
        refracted_origin = intersection.point + self.shadow_epsilon * refracted_direction.direction

        # Create the refracted ray
        refracted_ray = hc.Ray(refracted_origin, refracted_direction.direction)

        # Trace the refracted ray through the scene
        refracted_color = self.trace_ray(refracted_ray, depth + 1)

        # Return the color contributed by the refracted ray
        return refracted_color

    def render(self):
        image = np.zeros((self.width, self.height, 3))

        cam_dir = self.position - self.lookat
        d = 1.0
        top = d * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        for i in range(self.width):
            for j in range(self.height):
                colour = glm.vec3(0, 0, 0)
                if self.jitter:
                    for _ in range(self.samples):
                        ray = self.generate_ray(i, j, left, right, top, bottom, u, v, w, d)
                        colour += self.trace_ray(ray)
                    colour /= self.samples  # Average the color
                else:
                    ray = self.generate_ray(i, j, left, right, top, bottom, u, v, w, d)
                    colour += self.trace_ray(ray)
                # Clamp and assign the color to the image
                image[i, j, 0] = max(0.0, min(1.0, colour.x))
                image[i, j, 1] = max(0.0, min(1.0, colour.y))
                image[i, j, 2] = max(0.0, min(1.0, colour.z))
            print(i)
        return image
