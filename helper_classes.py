from cmath import inf
import numpy as np
from sqlalchemy import false, true


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# TODO:
# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
#  R = 2*(L*N)N - L
def reflected(vector, normal):
    norm_vector = normalize(vector)
    inner = np.dot(norm_vector, normal)
    inner = np.multiply(2 * inner, normal)
    return norm_vector - inner 

# # r=d−2(d⋅n)n
# def reflecter_v2(vector, normal):
#     vector_from_light = - vector
#     inner = np.dot(vector_from_light, normal)
#     inner = 2 * inner * vector
#     return vector_from_light - inner 


def reflecter_v2(vector, normal):
    norm_vector = (vector)
    inner = np.dot(norm_vector, normal)
    inner = 2 * inner * vector
    return inner  - norm_vector 


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

    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.direction))

    def get_distance_from_light(self, intersection):
        return np.inf

    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):

    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    def get_light_ray(self,intersection):
        return Ray(intersection,normalize(self.position - intersection))

    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):


    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(self.position, self.position - intersection)

    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        distance = self.get_distance_from_light(intersection)
        v_norm_point_to_light = normalize(self.position - intersection)
        attenuation_factor = self.kc + self.kl * distance + self.kq * distance * distance
        direction_norm_vector = normalize(self.direction)
        inner = np.inner(v_norm_point_to_light, direction_norm_vector)
        return (self.intensity * inner) / attenuation_factor

class Ray:
    def __init__(self, origin, direction, refraction=1.000273):
        self.origin = origin
        self.direction = direction
        # air refraction in index table
        self.refraction=refraction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects, current_object_that_ray_shoots=None):
        nearest_object = None
        min_distance = np.inf
        is_sphere = isinstance(current_object_that_ray_shoots, Sphere)
        if is_sphere:
            is_blocking = current_object_that_ray_shoots.checkifblocking(self)
            if is_blocking:
                min_distance = current_object_that_ray_shoots.intersect(self)
                nearest_object =  current_object_that_ray_shoots  
                return nearest_object, min_distance
        for obj in objects:
            is_mesh = isinstance(obj, Mesh)
            if current_object_that_ray_shoots and obj is current_object_that_ray_shoots:
                continue
            elif is_mesh:
                nearest_obj, curr_distance = obj.intersect(self)
                if curr_distance:
                    if curr_distance < min_distance:
                        min_distance = curr_distance
                        nearest_object = nearest_obj
            else:
                curr_distance = obj.intersect(self)
                if curr_distance:
                    if curr_distance < min_distance:
                        min_distance = curr_distance
                        nearest_object = obj
        return nearest_object, min_distance


class Object3D:

    def set_material(self, ambient, diffuse, specular, shininess, reflection, refraction=None):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.refraction = refraction


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
    
    def getOutwardFacingNormal(self, direction, intersectPoint=None):
        norm = self.normal
        point = self.point
        x,y,z = direction + point
        a,b,c = norm
        # d = np.abs(a*point[0] + b*point[1] + c*point[2]) / np.sqrt(a*a+b*b+c*c)
        d = a*point[0] + b*point[1] + c*point[2]
        if (a*x + b*y + c*z - d ) <= 0:
            return norm
        return -norm



class Triangle(Object3D):
    # Triangle gets 3 points as arguments
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    def compute_normal(self):
        v1 = self.c - self.a
        v2 = self.b - self.a
        return normalize(np.cross(v1, v2))
    
    def getOutwardFacingNormal(self, direction, intersectPoint=None):
        norm = self.normal
        point = self.a
        x,y,z = direction + point
        a,b,c = norm
        d = a*point[0] + b*point[1] + c*point[2]
        if (a*x + b*y + c*z - d ) <= 0:
            return norm
        return -norm
    # Hint: First find the intersection on the plane

    def get_triangle_plane(self):
        return Plane(self.normal, self.a)
    # Later, find if the point is in the triangle using barycentric coordinates
    def intersectv1(self, ray: Ray):
        # everything in this function was done according to slide 21, recitation 5
        ab, ac = self.b - self.a, self.c - self.a
        intersectionPoint = self.get_triangle_plane()
        intersectionPoint = intersectionPoint.intersect(ray)
        if not intersectionPoint:
            return None
        # areaABC = np.cross(ab,ac)
        # areaABC = normalize(areaABC) / 2
        areaABC = np.linalg.norm(np.cross((self.b - self.a), (self.c - self.a))) / 2
        PA,PB,PC = self.a - intersectionPoint, self.b - intersectionPoint,  self.c - intersectionPoint
        # alpha = np.cross(PB,PC)
        # alpha = normalize(alpha) / (2*areaABC)
        # beta = np.cross(PC,PA)
        # beta = normalize(beta)  / (2*areaABC)
        # gamma = 1 - alpha - beta
        alpha = np.linalg.norm(np.cross(PB, PC)) / (2 * areaABC)
        beta = np.linalg.norm(np.cross(PC, PA)) / (2 * areaABC)
        gamma = np.linalg.norm(np.cross(PA, PB)) / (2 * areaABC)
        
        if not np.abs(alpha + beta + gamma - 1) < 0.00001:
            return None
            
        return intersectionPoint

    def intersect(self, ray: Ray):
        # TODO
        plane = Plane(self.normal, self.a)
        intersection_distance= plane.intersect(ray)
        if intersection_distance == None:
            return None
        intersection_point = ray.origin + intersection_distance*ray.direction
        
        areaABC = np.linalg.norm(np.cross((self.b - self.a), (self.c - self.a))) / 2
        PA = self.a - intersection_point
        PB = self.b - intersection_point
        PC = self.c - intersection_point
        alpha = np.linalg.norm(np.cross(PB, PC)) / (2 * areaABC)
        beta = np.linalg.norm(np.cross(PC, PA)) / (2 * areaABC)
        gamma = np.linalg.norm(np.cross(PA, PB)) / (2 * areaABC)

        is_intersect = np.abs(alpha + beta + gamma - 1) < 0.00001
        if is_intersect:
            return intersection_distance
        else:
            return None


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        a = np.linalg.norm(ray.direction) ** 2
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta <= 0:
            return None
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        elif t1 > 0 or t2 > 0:
            return max(t1, t2)
        return None
    
    def getNormal(self, intersectPoint):
        norm = normalize(intersectPoint - self.center)
        return norm

    def getOutwardFacingNormal(self, direction, intersectPoint):
        norm = normalize(intersectPoint - self.center)
        plane = Plane(norm, intersectPoint)
        return plane.getOutwardFacingNormal(direction)


    def checkifblocking(self, ray: Ray):
        MARGIN=1e-5
        a = np.linalg.norm(ray.direction) ** 2
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta <= 0:
            return None
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 > 0 and t2 > 0:
            return True
        elif abs(t1) < MARGIN and t2 > 0:
            return True
        return False



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
        distance = np.inf
        nearest_obj = None
        for triangle in self.triangle_list:
            t = triangle.intersect(ray)
            if t :
                if t < distance:
                    distance = t
                    nearest_obj = triangle

        return nearest_obj, distance

