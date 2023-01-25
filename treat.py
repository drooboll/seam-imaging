from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

image = Image.open('banc-1.jpg').convert('L')

grad_x = ndimage.sobel(image, axis=0, mode='constant')
grad_y = ndimage.sobel(image, axis=1, mode='constant')

grad_l1 = np.abs(grad_x) + np.abs(grad_y)

print(image.width, image.height)

G = nx.DiGraph()

for width in range(image.width):
    for height in range(image.height):
        G.add_node(height * image.width + width, pos=(width, height))

source_index = image.width * image.height + 1
dest_index = image.width * image.height + 2
# Source 
G.add_node(source_index)
# Sink
G.add_node(dest_index)

back_weight = 10000

for width in range(1, image.width - 1, 1):
    print(f"Width {width} out of {image.width}")
    for height in range(image.height):
        G.add_edge(height * image.width + width, height * image.width + width + 1, capacity=grad_l1[height][width])
        G.add_edge(height * image.width + width + 1, height * image.width + width, capacity=back_weight)

        if height % 2 == 0 and height != image.height:
            G.add_edge(height * image.width + width + 1, (height + 1) * image.width + width, capacity=back_weight)
            G.add_edge((height + 1) * image.width + width + 1, height * image.width + width, capacity=back_weight)

max = np.max(grad_l1) * np.e

# connect first column
for height in range(image.height):
    G.add_edge(source_index, height * width, weight=max)

#connect last column
for height in range(image.height):
    G.add_edge((height + 1) * width, dest_index, weight=max)