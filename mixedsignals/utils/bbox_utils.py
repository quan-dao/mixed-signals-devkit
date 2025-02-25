from typing import List
import numpy as np
from mixedsignals.utils.geometry import make_se3, apply_se3_


def get_boxes_vertices_coord(boxes) -> List[np.ndarray]:
    """
            5-------- 1
           /|         /|
          4 -------- 0 .
          | |        | |
          . 6 -------- 2
          |/         |/
          7 -------- 3
        
           y
          /
         /
        /
        ==========> x
    """
    # box convention:
    # forward: 0 - 1 - 2 - 3, backward: 4 - 5 - 6 - 7, up: 0 - 1 - 5 - 4

    xs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float) / 2.0
    ys = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float) / 2.0
    zs = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float) / 2.0
    out = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        dx, dy, dz = box[3: 6].tolist()
        vers = np.concatenate([xs.reshape(-1, 1) * dx, ys.reshape(-1, 1) * dy, zs.reshape(-1, 1) * dz], axis=1)  # (8, 3)
        ref_se3_box = make_se3(box[:3], yaw=box[6])
        apply_se3_(ref_se3_box, points_=vers)
        out.append(vers)

    return out