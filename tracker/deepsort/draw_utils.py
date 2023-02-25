import matplotlib.pyplot as plt
from matplotlib import patches

def draw_pascal_voc_bboxes(
    plot_ax,
    bboxes,
    confidences = None,
):
    if confidences == None:
        confidences = [0 for i in range(len(bboxes))]

    #print("bboxes : ",bboxes)
    for i,bbox in enumerate(bboxes):
        top,left,width, height = bbox
        width = width - 2*top
        height = height - 2*left
        rect_1 = patches.Rectangle(
            (top,left),
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            (top,left),
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)

def show_image(image, bboxes,figsize=(10, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    if bboxes is not None:
        draw_pascal_voc_bboxes(ax, bboxes)
    plt.show()