from re import L
from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            color = np.zeros(3)
            # initializing ray
            ray = Ray(camera, normalize(pixel - camera))
            # find nearest object
            color += get_color(ray, lights, objects, ambient, 1, max_depth)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image
    
EPSILON = 1e-5
def get_color(ray, lights:LightSource, objects, ambient, level, max_level):
    color = np.zeros(3)
    nearest_object, min_distance = ray.nearest_intersected_object(objects)
    if not nearest_object:
        return color
    is_Sphere = isinstance(nearest_object, Sphere)
    intersection_point = calcIntersectPoint(ray,min_distance)
    direction = -lights[0].direction if hasattr(lights[0], "direction") else normalize(lights[0].position - intersection_point)
    if is_Sphere:
        outwardFacingNormal = normalize(nearest_object.getOutwardFacingNormal(direction, intersection_point))
    else:
        outwardFacingNormal = normalize(nearest_object.getOutwardFacingNormal(direction))
    intersection_point = intersection_point + EPSILON * outwardFacingNormal
    
    color += calc_emmited_color()
    color += calcAmbientColor(nearest_object, ambient) 
    for light in lights:
        ray_to_light = light.get_light_ray(intersection_point)
        _, min_distance_object_from_light = ray_to_light.nearest_intersected_object(objects)
        light_distance = light.get_distance_from_light(intersection_point)
        _, min_distance_object_from_light = ray_to_light.nearest_intersected_object(objects)

        if min_distance_object_from_light < light_distance and min_distance_object_from_light > 0.0001:
            return color
        color += calc_diffuse_color(nearest_object, light, intersection_point, is_Sphere)
        color += calc_specular_color(ray, nearest_object, light, intersection_point, is_Sphere)

        current_level = level + 1
        if current_level > max_level:
            return color

        reflected_vector = reflected(normalize(ray.direction), outwardFacingNormal)
        reflactive_ray = Ray(intersection_point, reflected_vector)
        color += np.multiply(nearest_object.reflection, get_color(reflactive_ray, lights, objects, ambient, current_level, max_level))
    
    return color

def calc_specular_color(ray:Ray, nearest_object:Object3D, light:LightSource, intersection_point, is_Sphere=False):
    to_light_vector = normalize(light.get_light_ray(intersection_point).direction)
    if is_Sphere:
        reflected_vector = (reflected(to_light_vector,  nearest_object.getOutwardFacingNormal(to_light_vector, -to_light_vector))) 
    else:
        reflected_vector = (reflected(to_light_vector,  nearest_object.getOutwardFacingNormal(to_light_vector))) 
    v = normalize(ray.direction)
    inner = np.inner(v, reflected_vector)
    inner = np.power(inner, nearest_object.shininess)
    return nearest_object.specular * light.get_intensity(intersection_point) * inner

def calc_diffuse_color(nearest_object:Object3D, light:LightSource, intersection_point, is_Sphere=False):
    norm_light_to_point_vector = normalize(-light.get_light_ray(intersection_point).direction)
    if is_Sphere:
        inner = np.inner(nearest_object.getOutwardFacingNormal(-norm_light_to_point_vector, intersection_point), norm_light_to_point_vector)
    else:
        inner = np.inner(nearest_object.getOutwardFacingNormal(-norm_light_to_point_vector), norm_light_to_point_vector)

    return nearest_object.diffuse * light.get_intensity(intersection_point) * inner


def calcAmbientColor(nearest_object:Object3D, ambient):
    return nearest_object.ambient *ambient

def calc_emmited_color():
    return 0

def calcIntersectPoint(ray:Ray, distance):
    return ray.origin + ray.direction * distance

# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects

