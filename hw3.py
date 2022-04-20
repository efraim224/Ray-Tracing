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
def get_color(ray: Ray, lights:LightSource, objects, ambient, level, max_level, object_from_ray=None):
    color = np.zeros(3)
    nearest_object, min_distance = ray.nearest_intersected_object(objects, object_from_ray)
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
        not_blocked_by_light = is_light_blocked_by_object(light, objects, nearest_object, intersection_point)

        if not not_blocked_by_light:
            color += calc_diffuse_color(nearest_object, light, intersection_point, is_Sphere)
            color += calc_specular_color(ray, nearest_object, light, intersection_point, is_Sphere)

    current_level = level + 1
    if current_level > max_level:
        return color

    reflected_vector = reflected(normalize(ray.direction), outwardFacingNormal)
    reflactive_ray = Ray(intersection_point, reflected_vector)
    color += np.multiply(nearest_object.reflection, get_color(reflactive_ray, lights, objects, ambient, current_level, max_level, nearest_object))
    
    return color

def is_light_blocked_by_object(light, objects, nearest_object, intersection_point):
    not_blocked_by_light = False
    ray_to_light = light.get_light_ray(intersection_point)
    ray_light_obj, min_distance_object_from_light = ray_to_light.nearest_intersected_object(objects, nearest_object)
    ray_light_obj_is_Sphere = isinstance(ray_light_obj, Sphere)
    if ray_light_obj_is_Sphere:
        sphere_is_blocking_light = ray_light_obj.checkifblocking(ray_to_light)
        if sphere_is_blocking_light:
            not_blocked_by_light = True
    light_distance = light.get_distance_from_light(intersection_point)
    if min_distance_object_from_light < light_distance and min_distance_object_from_light > 0.0001:
        not_blocked_by_light = True
    return not_blocked_by_light

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
    is_spotlight = isinstance(light, SpotLight)
    if is_spotlight:
        norm_light_to_point_vector = normalize(light.get_light_ray(intersection_point).direction)
    else:
        norm_light_to_point_vector = normalize(-light.get_light_ray(intersection_point).direction)
    if is_Sphere:
        inner = np.inner(nearest_object.getOutwardFacingNormal(-norm_light_to_point_vector, intersection_point), norm_light_to_point_vector)
    else:
        if is_spotlight:
            inner = np.inner(nearest_object.getOutwardFacingNormal(-norm_light_to_point_vector), norm_light_to_point_vector)
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
    sphere_a = Sphere([-0.5, 0.2, -1],0.5)
    sphere_a.set_material([0, 0.1, 1], [0, 0.1, 1], [0.3, 0.3, 0.3], 100, 0.2)
    # b: (x+0.3)^(2)+(y-0.99)^(2)+(z+1)^(2)-0.1=0
    sphere_b = Sphere([0.3, -0.99, -1], 0.3162)
    sphere_b.set_material([0.1, 1, 0], [0.1, 1, 0], [0.3, 0.3, 0.3], 100, 0.2)
    # eq2: (x+0.12)^(2)+(y-1.47)^(2)+(z+1)^(2)-0.04=0
    shepre_c = Sphere([0.1, 1, -1], 0.2)
    shepre_c.set_material([1, 0, 0.1], [1, 0, 0.1], [0.3, 0.3, 0.3], 100, 0.2)
    
    background = Plane(normal=[0,0,1], point=[0,0,-2])
    background.set_material([0.5, 0.5, 0.3], [0.7, 0.7, 1], [1, 1, 1], 1000, 0.5)
    
    light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([1,1.5,1]),kc=0.1,kl=0.1,kq=0.1)

    lights = [light]
    objects = [sphere_a, sphere_b, shepre_c, background]
    return camera, lights, objects

