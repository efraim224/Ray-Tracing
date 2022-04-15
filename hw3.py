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
            ray = Ray(camera, pixel - camera)
            nearest_object, min_distance = ray.nearest_intersected_object(objects)
            intersection_point = calcIntersectPoint(ray,min_distance)
            color += get_color(ray, lights,nearest_object, ambient, intersection_point)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image

def get_color(ray, lights:LightSource, nearest_object, ambient, intersection_point):
    color = np.zeros(3)
    color += calc_emmited_color()
    color += calcAmbientColor(nearest_object, ambient)            
    for light in lights:
        color += calc_diffuse_color(nearest_object, light, intersection_point)
        color += calc_specular_color(ray, nearest_object, light, intersection_point)
    return color

def calc_specular_color(ray:Ray, nearest_object:Object3D, light:LightSource, intersection_point):
    # to_light_vector = normalize(light.position - intersection_point)
    to_light_vector = light.position - intersection_point
    reflected_vector =  reflected(to_light_vector , nearest_object.normal)
    v = ray.direction
    inner = np.inner(v, reflected_vector)
    inner = np.power(inner, nearest_object.shininess)
    return nearest_object.specular * light.get_intensity(intersection_point) * inner

def calc_diffuse_color(nearest_object:Object3D, light:LightSource, intersection_point):
    norm_light_to_point_vector = normalize(light.position - intersection_point)
    inner = np.inner(nearest_object.normal, norm_light_to_point_vector)

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

