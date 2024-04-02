import math

import glm
import numpy as np

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
                 ambient: glm.vec3,
                 lights: list[hc.Light],
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
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.materials = materials  # all materials of objects in the scene
        self.objects = objects  # all objects in the scene
        self.shadow_epsilon = 1e-3
    
    def generate_ray(self, i, j, left, right, top, bottom, u, v, w, d):
        u_coord = left + (right - left) * (i + 0.5) / self.width
        v_coord = bottom + (top - bottom) * (j + 0.5) / self.height
        ray_dir = glm.normalize(u * u_coord + v * v_coord - w * d)
        return hc.Ray(self.position, ray_dir)
    
    def find_closest_intersection(self, ray):
        closest_t = float('inf')
        closest_intersection = hc.Intersection.default()
        for obj in self.objects:
            intersection = hc.Intersection.default()
            if obj.intersect(ray, intersection) and intersection.time < closest_t:
                closest_t = intersection.time
                closest_intersection = intersection
        return closest_intersection
    
    def calculate_light_contribution(self, intersection, light):
        colour = glm.vec3(0,0,0)

        # Ambient
        colour += self.ambient * light.colour

        # Diffuse
        L = glm.normalize(light.vector - intersection.point)
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

    
    def is_in_shadow(self, intersection, light):
        offset_position = intersection.point + intersection.normal * self.shadow_epsilon
        to_light = light.vector - offset_position
        shadow_ray = hc.Ray(offset_position, to_light)
        for obj in self.objects:
            if obj.intersect(shadow_ray, hc.Intersection.default()):
                return True
        return False

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
                # Generate Rays
                ray = self.generate_ray(i, j, left, right, top, bottom, u, v, w, d)

                # Test for intersection
                closest_intersection = self.find_closest_intersection(ray)

                # TODO: Perform shading computations on the intersection point
                if closest_intersection.time < float('inf'):
                    colour = glm.vec3(0, 0, 0)
                    for light in self.lights:
                        if self.is_in_shadow(closest_intersection, light):
                            colour += self.ambient * light.colour
                        else:
                            colour += self.calculate_light_contribution(closest_intersection, light)
                    colour = glm.clamp(colour, 0, 1)
                else:
                    colour = glm.vec3(0, 0, 0)

                image[i, j, 0] = max(0.0, min(1.0, colour.x))
                image[i, j, 1] = max(0.0, min(1.0, colour.y))
                image[i, j, 2] = max(0.0, min(1.0, colour.z))

        return image
