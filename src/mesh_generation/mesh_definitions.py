"""
This codebase uses the following convention for edge orientations
for 2D elements.
       E1
    [v1 → v2]
 E2 [↓    ↓] E3
    [v3 → v4]
       E4
We use the cubed-sphere convention
              --------------
              |            |
              |  (π/2, ·)  |
              |y     ·     |
              |↑    Top    |
              |·→x         |
  ----------------------------------------------------
  |           |            |            |            |
  | (0, 3π/2) | (ϕ=0, λ=0) |  (0, π/2)  |   (0, π)   |
  |y    ·     |y     ·     |y     ·     |y     ·     |
  |↑  Left    |↑   Front   |↑   Right   |↑   Back    |
  |·→x        |·→x         |·→x         |·→x         |
  ----------------------------------------------------
              |            |
              |  (-π/2, ·) |
              |y     ·     |
              |↑  Bottom   |
              |·→x         |
              --------------
where the above diagram is folded along non-conforming edges
to form a cube.
"""
import numpy as np
BACKWARDS, FORWARDS = (0, 1)
TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE = (0, 1, 2, 3, 4, 5)
TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE = (0, 1, 2, 3)


face_topo = {TOP_FACE: {TOP_EDGE: (BACK_FACE, TOP_EDGE, BACKWARDS),
                        LEFT_EDGE: (LEFT_FACE, TOP_EDGE, FORWARDS),
                        RIGHT_EDGE: (RIGHT_FACE, TOP_EDGE, BACKWARDS),
                        BOTTOM_EDGE: (FRONT_FACE, TOP_EDGE, FORWARDS)},
             BOTTOM_FACE: {TOP_EDGE: (FRONT_FACE, BOTTOM_EDGE, FORWARDS),
                           LEFT_EDGE: (LEFT_FACE, BOTTOM_EDGE, BACKWARDS),
                           RIGHT_EDGE: (RIGHT_FACE, BOTTOM_EDGE, FORWARDS),
                           BOTTOM_EDGE: (BACK_FACE, BOTTOM_EDGE, BACKWARDS)},
             FRONT_FACE: {TOP_EDGE: (TOP_FACE, BOTTOM_EDGE, FORWARDS),
                          LEFT_EDGE: (LEFT_FACE, RIGHT_EDGE, FORWARDS),
                          RIGHT_EDGE: (RIGHT_FACE, LEFT_EDGE, FORWARDS),
                          BOTTOM_EDGE: (BOTTOM_FACE, TOP_EDGE, FORWARDS)},
             BACK_FACE: {TOP_EDGE: (TOP_FACE, TOP_EDGE, BACKWARDS),
                         LEFT_EDGE: (RIGHT_FACE, RIGHT_EDGE, FORWARDS),
                         RIGHT_EDGE: (LEFT_FACE, LEFT_EDGE, FORWARDS),
                         BOTTOM_EDGE: (BOTTOM_FACE, BOTTOM_EDGE, BACKWARDS)},
             LEFT_FACE: {TOP_EDGE: (TOP_FACE, LEFT_EDGE, FORWARDS),
                         LEFT_EDGE: (BACK_FACE, RIGHT_EDGE, FORWARDS),
                         RIGHT_EDGE: (FRONT_FACE, LEFT_EDGE, FORWARDS),
                         BOTTOM_EDGE: (BOTTOM_FACE, LEFT_EDGE, BACKWARDS)},
             RIGHT_FACE: {TOP_EDGE: (TOP_FACE, RIGHT_EDGE, BACKWARDS),
                          LEFT_EDGE: (FRONT_FACE, RIGHT_EDGE, FORWARDS),
                          RIGHT_EDGE: (BACK_FACE, LEFT_EDGE, FORWARDS),
                          BOTTOM_EDGE: (BOTTOM_FACE, RIGHT_EDGE, FORWARDS)}}
verts = [[-1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [-1.0, -1.0, 1.0],
         [1.0, -1.0, 1.0],
         [-1.0, 1.0, -1.0],
         [1.0, 1.0, -1.0],
         [-1.0, -1.0, -1.0],
         [1.0, -1.0, -1.0]]
vert_info = {TOP_FACE: np.array([verts[2], verts[3], verts[0], verts[1]]),
             BOTTOM_FACE: np.array([verts[4], verts[5], verts[6], verts[7]]),
             FRONT_FACE: np.array([verts[0], verts[1], verts[4], verts[5]]),
             BACK_FACE: np.array([verts[3], verts[2], verts[7], verts[6]]),
             LEFT_FACE: np.array([verts[2], verts[0], verts[6], verts[4]]),
             RIGHT_FACE: np.array([verts[1], verts[3], verts[5], verts[7]])}
axis_info = {TOP_FACE: (0, 1.0, 1, -1.0),
             BOTTOM_FACE: (0, 1.0, 1, 1.0),
             FRONT_FACE: (0, 1.0, 2, 1.0),
             BACK_FACE: (0, -1.0, 2, 1.0),
             LEFT_FACE: (1, 1.0, 2, 1.0),
             RIGHT_FACE: (1, -1.0, 2, 1.0)}

# NOTE: this may not be appropriate for unstructured grids!
MAX_VERT_DEGREE = 4
MAX_VERT_DEGREE_UNSTRUCTURED = 8
