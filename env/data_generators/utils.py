import random
from shapely.geometry import Polygon, Point

def get_random_point_in_polygon(poly):
     minx, miny, maxx, maxy = poly.bounds
     while True:
         p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         if poly.contains(p):
             return p

p = Polygon([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])
point_in_poly = get_random_point_in_polygon(p)

x_coords = []
y_coords = []

for i in range(10000):
    point = get_random_point_in_polygon(p)
    x_coords.append(point.x)
    y_coords.append(point.y)

print("haqeesdf")

import seaborn
import matplotlib.pyplot as plt

seaborn.set(style='whitegrid')

seaborn.scatterplot(x=x_coords,
                    y=y_coords)

plt.show()