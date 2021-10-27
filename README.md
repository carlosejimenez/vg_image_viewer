## VG-Image-Viewer

This module provides the NotebookImageViewer class to display images, with or without scene-graph bounding boxes.

Example usage in Jupyert Notebook or Jupyter Lab:
```python
from vg_image_viewer import NotebookImageViewer

viewer = NotebookImageViewer('VG_images/', scene_graphs='scene_graphs.json')
viewer.show_image('21', with_bboxes=True)
```