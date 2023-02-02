from PIL import Image
from skimage import img_as_float
from skimage.segmentation import chan_vese
from scipy import ndimage
import numpy as np
import graph_tool.all as gt
from datetime import datetime
from fractions import Fraction
import math
import sys

import matplotlib.pyplot as plt

# Part of contour-based energy
alpha = 0.5

# Chan-Vese parameters
mu = 0.016
lambda1 = 1.25
lambda2 = 0.75
tolerance=2e-3


def get_node_index(image, x, y):
    return y * image.width + x


def get_gradient_image(image):
    grad_x = ndimage.sobel(image, axis=0, mode='constant')
    grad_y = ndimage.sobel(image, axis=1, mode='constant')


    return np.abs(grad_x) + np.abs(grad_y)


def get_contour_image(image):
    image_flt = img_as_float(image)

    # seems logical for me, since homogenous clusters normally should be with small gradient -> same segment
    init = get_gradient_image(image) / 2000

    # parameters are set to work ok with "plage" and "ouiseau"
    segmentation, _, en = chan_vese(image_flt, mu=mu, lambda1=lambda1, lambda2=lambda2, tol=tolerance,
               max_num_iter=100, dt=0.5, init_level_set=init,
               extended_output=True)

    # it's called cheating
    # in fact the problem is caused by low-contrast images,
    # and algo sometimes swap object and background,
    # so we have to invert it...
    if np.sum(segmentation) > 0.5 * image.width * image.height:
        segmentation = 1 - segmentation

    segmentation = (get_gradient_image(segmentation) > 0)

    return Image.fromarray((segmentation * 255).astype(np.uint8))


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

def cut_image_x(image, contour):
    out_image = Image.new("L", (image.width - 1, image.height))
    new_contour = Image.new("L", (image.width - 1, image.height))
    print(datetime.now(), "Graph build")
    G = build_graph(image)
    grad = get_gradient_image(image)

    
    energies = alpha * np.array(contour) + (1 - alpha) * grad
    
    G = graph_add_weights(G, energies)
    print(datetime.now(), "Graph build end")
    border = graph_get_border(G)

    for width in range(image.width):
        for height in range(image.height):
            if width == border[height]:
                continue

            dest_height = height
            dest_width = width if width < border[height] else width - 1
            out_image.putpixel((dest_width, dest_height), image.getpixel((width, height)))
            new_contour.putpixel((dest_width, dest_height), contour.getpixel((width, height)))
    
    return out_image, new_contour

# Parameters
# 1 - name of the image to crop
# 2 - crop on aX in pixels
# 3 - crop on aY in pixels

if __name__ == '__main__':
    filename = sys.argv[1]
    steps_x = int(sys.argv[2])
    steps_y = int(sys.argv[3])

    image = Image.open(filename).convert('L')
    contour = get_contour_image(image)

    plt.imshow(contour)
    plt.show()

    print(f"Cropping {filename} to {image.width - steps_x}x{image.height - steps_y}")

    start_time = datetime.now()
    operations = []

    # Here goes a bit weird step sequence.
    # The idea is to do as much interleaves of x and y axis seaming

    if steps_y == 0:
        if steps_x == 0:
            exit(0)
        operations = 'x' * steps_x

    elif steps_x == 0:
        operations = 'y' * steps_y
    
    else:
        # Not the best approach, but it should work
        # in fact it is better to do xyxyxy... 
        frac = Fraction(round(steps_x / steps_y, 1)).limit_denominator()
        x_series = math.floor(steps_x / frac.numerator)
        x_left = steps_x - x_series * frac.numerator
        y_series = math.floor(steps_y / frac.denominator)
        y_left = steps_y - x_series * frac.denominator
        operations = ('x' * frac.numerator + 'y' * frac.denominator) * x_series
        operations += 'x' * x_left + 'y' * y_left

        print(f"Steps x:{ frac.numerator}, steps y: {frac.denominator}, series: {x_series}, {y_series}")
        print(f"Steps left: {x_left}, {y_left}")
        print(operations)

    count = 0
    for op in operations:
        # TODO: do not need to rotate for consecutive 'y'-s actually
        if op == 'y':
            image = image.rotate(90, expand=True)
            contour = contour.rotate(90, expand=True)
        image, contour = cut_image_x(image, contour)
        if op == 'y':
            image = image.rotate(270, expand=True)
            contour = contour.rotate(270, expand=True)
        count += 1
        image.save(f"out/processing-{count}.jpg")
        contour.save(f"out/contour-{count}.jpg")

    print(datetime.now(), "--- %s seconds ---" % (datetime.now() - start_time))

    image.save("test-out.jpg")
