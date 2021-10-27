import json
import random
from pathlib import Path

import cv2
import numpy as np
from IPython.display import display
from PIL import Image


def draw_rectangle(img: np.ndarray,
                   bbox: list,
                   bbox_color: tuple = (255, 255, 255),
                   thickness: int = 3,
                   is_opaque: bool = False,
                   alpha: float = 0.5) -> np.ndarray:
    r"""Draws the rectangle around the object.

    Args:
        img: the actual image.
        bbox: a list containing x_min, y_min, x_max and y_max of the rectangle positions.
        bbox_color: the color of the box, by default (255,255,255).
        thickness: thickness of the outline of the box, by default 3.
        is_opaque: if False, draws a solid rectangular outline. Else, a filled rectangle which is semi transparent, by
            default False.
        alpha: strength of the opacity, by default 0.5.

    Returns:
        The image with the bounding box drawn.
    """

    output = img.copy()
    if not is_opaque:
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, thickness)
    else:
        overlay = img.copy()

        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def add_label(img,
              label,
              bbox,
              draw_bg=True,
              text_bg_color=(255, 255, 255),
              text_color=(0, 0, 0),
              top=True) -> np.ndarray:
    r"""Adds label, inside or outside the rectangle.

    Args:
        img: the image on which the label is to be written, preferably the image with the rectangular bounding box
            drawn.
        label: the text (label) to be written.
        bbox: a list containing x_min, y_min, x_max and y_max of the rectangle positions.
        draw_bg: if True, draws the background of the text, else just the text is written, by default True.
        text_bg_color: the background color of the label that is filled, by default (255, 255, 255).
        text_color: color of the text (label) to be written, by default (0, 0, 0).
        top: if True, writes the label on top of the bounding box, else inside, by default True.

    Returns:
        The image with the label written.
    """
    pixel_height = 10
    five = 0
    label_width_offset = 5
    font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_PLAIN, pixel_height)
    (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, 2)[0]

    if top:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width - label_width_offset, bbox[1] - text_height]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + five, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + five, bbox[1] - five),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 2)

    else:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width - label_width_offset, bbox[1] + text_height]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + five, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + five, bbox[1] - five + text_height),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 2)

    return img


def add_multiple_labels(img: np.ndarray,
                        labels: list,
                        bboxes: list,
                        draw_bg: bool = True,
                        text_bg_color: tuple = (255, 255, 255),
                        thickness: int = 2,
                        is_opaque: bool = False,
                        alpha: float = 0.1,
                        top: bool = True) -> np.ndarray:
    r"""Add labels, inside or outside the rectangles.

    Args:
        img: the image on which the labels are to be written, preferably the image with the rectangular bounding boxes
            drawn.
        labels: a list of string of the texts (labels) to be written.
        bboxes: a list of lists, each inner list containing x_min, y_min, x_max and y_max of the rectangle positions.
        draw_bg: if True, draws the background of the texts, else just the texts are written, by default True.
        text_bg_color: the background color of the labels that are filled, by default (255, 255, 255).
        thickness: thickness of bounding box lines.
        is_opaque: if True, bounding box line is opaque rather than transparent with alpha.
        alpha: if is_opaque if False, alpha is the transparency of the drawn bounding boxes.
        top: if True, writes the labels on top of the bounding boxes, else inside, by default True.

    Returns:
        The image with the labels written.
    """

    colors = list()

    for bbox in bboxes:
        # cycle through some distinct colors
        color = random.choice([(255, 0, 155), (0, 188, 255), (255, 239, 0), (0, 0, 255)])
        colors.append(color)
        img = draw_rectangle(img, bbox, color, thickness, is_opaque,
                             alpha)

    for label, bbox, color in zip(labels, bboxes, colors):
        img = add_label(img, label, bbox, draw_bg, text_bg_color, color, top)

    return img


class NotebookImageViewer:
    r"""Can be used to display images (using an id as filename from supplied the root directory) in jupyter notebooks
     and lab.

     Args:
         root: root directory where images should be placed by default.
         default_dim: by default images will be normalized to fit within the size default_dim x default_dim before
            displaying.
         scene_graphs: dictionary with id -> scene_graph **or** filepath where such a json dictionary exists.
    """

    def __init__(self, root: str, default_dim: int = 500, scene_graphs: [str, dict] = None):
        # below will throw does not exist error
        self.root = Path(root).resolve(strict=True)
        assert self.root.is_dir(), NotADirectoryError('The passed root %s is not a directory.' % self.root)
        self.default_dim = default_dim
        assert default_dim > 0, ValueError('default_dim must be positive!')
        self.scene_graphs = scene_graphs if type(scene_graphs) is dict else json.load(open(scene_graphs))

    @staticmethod
    def _extract_label_and_bbox(entry: dict, ratio) -> tuple:
        r"""Converts bbox entry to (name, (x, y, x+w, y+h)) for adding to the image.

        Args:
            entry: object entry from scene graph.
        """
        return str(entry['name']), tuple(map(lambda x: int(round(x * ratio)),
                                             (entry['x'], entry['y'], entry['x']+entry['w'], entry['y']+entry['h'])))

    def show_image(self, value: [str, int], ext: str = '.jpg', with_bboxes: bool = False):
        r"""Displays image with name value in root directory.

        Args:
            value: name of file in root directory (without extension).
            ext: extension to use for value, so file reads {root}/{value}{ext}. '.jpg' by default.
            with_bboxes: if scene_graphs contains value, will generate image with bounding boxes.
        """
        img_filename = Path(self.root.as_posix(), str(value) + ext).as_posix()
        img = Image.open(img_filename)
        init_scale = 600
        ratio = init_scale / max(img.size[0], img.size[1])
        img = img.resize([round(img.size[0] * ratio), round(img.size[1] * ratio)])
        if with_bboxes:
            objects = self.scene_graphs[str(value)]['objects'].values()
            labels, bboxes = zip(*list(map(lambda x: self._extract_label_and_bbox(x, ratio), objects)))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = add_multiple_labels(img, labels, bboxes, draw_bg=True, text_bg_color=(255, 255, 255), thickness=2,
                                     is_opaque=False, alpha=0.1, top=False)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        ratio = self.default_dim / init_scale
        img = img.resize([round(img.size[0] * ratio), round(img.size[1] * ratio)])
        display(img)
