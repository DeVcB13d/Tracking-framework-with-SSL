# Plotting  functions
import matplotlib.pyplot as plt
from matplotlib import patches
def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right
    
    return bottom_left, width, height
def get_pascal_bbox_list(bbox):
    # COCO bbox format : x y width height
    xmin,ymin = bbox[0],bbox[1]
    width,height = bbox[2],bbox[3]
    xmax = xmin + width
    ymax = ymin + height
    return [xmin,ymin,xmax,ymax]
def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    #print("bboxes : ",bboxes)
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)

def show_image(
    image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()

'''
Function to check if a bbox is valid or not
'''
def check_bboxes(bbox):
    # Check if there are incompatible bboxes
    if bbox[0] == bbox[2]:
        return False
    elif bbox[1] == bbox[3]:
        return False
    else:
        return True