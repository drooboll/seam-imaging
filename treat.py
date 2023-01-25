from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

image = Image.open('plage.jpg').convert('L')

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

back_weight = np.sum(grad_l1)

for width in range(1, image.width - 1, 1):
    print(f"Width {width} out of {image.width}")
    for height in range(image.height):
        G.add_edge(height * image.width + width, height * image.width + width + 1, capacity=grad_l1[height][width])
        G.add_edge(height * image.width + width + 1, height * image.width + width, capacity=back_weight)

        if height % 2 == 0 and height != image.height:
            G.add_edge(height * image.width + width + 1, (height + 1) * image.width + width, capacity=back_weight)
            G.add_edge((height + 1) * image.width + width + 1, height * image.width + width, capacity=back_weight)

# connect first column
for height in range(image.height):
    G.add_edge(source_index, height * width, capacity=back_weight)

#connect last column
for height in range(image.height):
    G.add_edge((height + 1) * width, dest_index, capacity=back_weight)


cut_value, partition = nx.minimum_cut(G, source_index, dest_index, capacity='capacity')

# border calculation
edge_nodes = []
for p1_node in partition[0]:
    for p2_node in partition[1]:
        if G.has_edge(p1_node, p2_node):
            # taking left part
            edge_nodes.append(p1_node)

print(edge_nodes)

for node in edge_nodes:
    x = node % image.width
    y = math.floor(node / image.width)
    print(x, y)
    image.putpixel((x, y), (255))
