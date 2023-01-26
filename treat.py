from PIL import Image
from scipy import ndimage
import numpy as np
import graph_tool.all as gt
import graph_tool.draw as gt_draw
from datetime import datetime, timedelta
import math
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def get_node_index(image, x, y):
    return y * image.width + x

def get_gradient_image(image):
    grad_x = ndimage.sobel(image, axis=0, mode='constant')
    grad_y = ndimage.sobel(image, axis=1, mode='constant')

    return np.abs(grad_x) + np.abs(grad_y)

def build_graph(image):
    print(datetime.now(), "Size:", image.width, image.height)

    G = gt.Graph()
    G.add_vertex(image.width * image.height + 2)

    return G

def graph_add_weights(G, grad_l1):
    back_weight = np.sum(grad_l1)

    edge_list = []
    for width in range(image.width):
        for height in range(image.height):

            current = G.vertex(get_node_index(image, width, height))
            right = G.vertex(get_node_index(image, width + 1, height))

            if width != image.width - 1: 
                edge_list.append(
                    (
                        current,
                        right,
                        grad_l1[height][width]
                    )
                )
                edge_list.append(
                    (
                        right,
                        current,
                        back_weight
                    )
                )

            if width != 0 and height != image.height - 1:
                edge_list.append(
                    (
                        current,
                        G.vertex(get_node_index(image, width - 1, height + 1)),
                        back_weight
                    )
                )

            if width != image.width - 1 and height != image.height - 1:
                edge_list.append(
                    (
                        G.vertex(get_node_index(image, width + 1, height + 1)),
                        current,
                        back_weight
                    )
                )

    source = G.vertex(get_node_index(image, 0, image.height))
    sink = G.vertex(get_node_index(image, 1, image.height))
    
    for height in range(image.height):
        # connect first column
        edge_list.append(
            (
                source,
                G.vertex(get_node_index(image, 0, height)),
                back_weight
            )
        )

        #connect last column
        edge_list.append(
            (
                G.vertex(get_node_index(image, image.width - 1, height)),
                sink,
                back_weight
            )
        )

    weight = G.new_edge_property("int")
    eprops = [weight]
    G.add_edge_list(edge_list, eprops=eprops)
    G.edge_properties["weight"] = weight

    return G


def graph_get_border(G):
    print(datetime.now(), "max flow calc")

    res = gt.push_relabel_max_flow(
    #res = gt.boykov_kolmogorov_max_flow(
        G,
        G.vertex(get_node_index(image, 0, image.height)),
        G.vertex(get_node_index(image, 1, image.height)),
        G.edge_properties["weight"]
    )

    print(datetime.now(), "min cut calc")
    partition = gt.min_st_cut(
        G,
        G.vertex(get_node_index(image, 0, image.height)),
        G.edge_properties["weight"],
        res
    )

    print(datetime.now(), "min cut calc finished")
    # border calculation

    cap = G.edge_properties["weight"]

    """
    res.a = cap.a - res.a 
    edge_pen_width=gt.prop_to_size(res, mi=1, ma=5, power=0.5)
    pos = G.new_vertex_property("vector<double>")
    vprop = {}

    for v in G.vertices():
        if G.vertex_index[v] == image.width * image.height:
            pos[v] = (0, 10*image.height / 2)
            continue
        if G.vertex_index[v] == image.width * image.height + 1:
            pos[v] = (10*image.width + 20, 10*image.height / 2)
            continue
        pos[v] = (G.vertex_index[v] % image.width*10 + 10, 10*math.floor(G.vertex_index[v] / image.width))
        vprop[v] = f"{G.vertex_index}"

    G.vertex_properties["pos"] = pos

    gt_draw.graphviz_draw(G, pos=pos, vprops=vprop, pin=True, output="test.png", size=(40,40), vsize=4, penwidth=edge_pen_width, vcolor=partition)
    """

    left = []
    border = []

    print(datetime.now(), "sorting")

    for vertex in G.vertices():
        if partition[vertex]:
            left.append(G.vertex(vertex))
    left_sorted = sorted(left, key=lambda node: math.floor(G.vertex_index[node] / image.width))
    print(datetime.now(), "sorted")

    for vertex in left_sorted: 
        node = G.vertex_index[vertex]
        x = node % image.width
        y = math.floor(node / image.width)

        if len(border) == 0:
            border.append(node)
            continue

        x_last = border[-1] % image.width
        y_last = math.floor(border[-1] / image.width)

        if y >= image.height:
            continue
            
        # exchange
        if y_last == y:
            if x_last < x:
                border[-1] = node
            continue

        border.append(node)

    return [x % image.width for x in border]

def cut_image_x(image):
    image = image.convert('L')
    out_image = Image.new("L", (image.width - 1, image.height))
    print(datetime.now(), "Graph build")
    G = build_graph(image)
    grad = get_gradient_image(image)
    G = graph_add_weights(G, grad)
    print(datetime.now(), "Graph build end")
    border = graph_get_border(G)

    for width in range(image.width):
        for height in range(image.height):
            if width == border[height]:
                continue

            dest_height = height
            dest_width = width if width < border[height] else width - 1
            #print(datetime.now(), out_image.width, out_image.height, dest_width, dest_height)
            out_image.putpixel((dest_width, dest_height), image.getpixel((width, height)))
    
    return out_image


image = Image.open('banc.jpg')

start_time = datetime.now()
for i in range(50):
    image = cut_image_x(image)
    image.save(f"out/processing-{i}.jpg")

print(datetime.now(), "--- %s seconds ---" % (datetime.now() - start_time))

image.save("test-out.jpg")
