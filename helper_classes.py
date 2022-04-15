import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# TODO:
# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    norm_vector = normalize(vector)
    inner = np.dot(norm_vector, normal)
    inner = 2 * inner * normal
    return inner - norm_vector




## Lights



class LightSource:

    def __init__(self, intensity):
        self.intensity = intensity
    def get_light_ray(self,intersection_point):
        pass
    def get_distance_from_light(self, intersection):
        pass
    def get_intensity(self, intersection):
        pass

class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = np.array(direction)


    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection,normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):

    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the point to a light source
    def get_light_ray(self,intersection):
        return Ray(intersection,normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):


    def __init__(self, intensity):
        super().__init__(intensity)
        # TODO

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        # TODO
        return Ray()

    def get_distance_from_light(self,intersection):
        #TODO
        pass

    def get_intensity(self, intersection):
        #TODO
        pass


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for obj in objects:
            curr_distance = obj.intersect(self)
            if curr_distance:
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    nearest_object = obj
        return nearest_object, min_distance


class Object3D:

    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t
        else:
            return None



class Triangle(Object3D):
    # Triangle gets 3 points as arguments
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    def compute_normal(self):
        v1 = self.c - self.a
        v2 = self.c - self.b
        return np.cross(v1, v2)

    def get_triangle_plane(self):
        return Plane(self.compute_normal, self.a)

    # Hint: First find the intersection on the plane
    # Later, find if the point is in the triangle using barycentric coordinates
    def intersect(self, ray: Ray):
        ab = self.a - self.b
        
        pass


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        #TODO
        pass


class Mesh(Object3D):
    # Mesh are defined by a list of vertices, and a list of faces.
    # The faces are triplets of vertices by their index number.
    def __init__(self, v_list, f_list):
        self.v_list = v_list
        self.f_list = f_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        triangle_list = []
        for triangle_points in self.f_list:
            a = self.v_list[triangle_points[0]]
            b = self.v_list[triangle_points[1]]
            c = self.v_list[triangle_points[2]]
            triangle_list.append(Triangle(a, b ,c))
        return triangle_list

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        distance = np.infi
        nearest_obj = None
        for triangle in self.triangle_list:
            t = triangle.intersect(ray)
            if t :
                if t < distance:
                    distance = t
                    nearest_obj = triangle

        return nearest_obj, triangle