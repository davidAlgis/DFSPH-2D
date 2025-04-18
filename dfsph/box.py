import numpy as np
from numba import float32, int32, njit
from numba.experimental import jitclass

# Define the Box class specification
box_spec = [
    ("origin_x", float32),
    ("origin_y", float32),
    ("size_x", float32),
    ("size_y", float32),
]


def box_repr(box):
    return (
        f"Box(origin_x={box.origin_x}, origin_y={box.origin_y},"
        f"size_x={box.size_x}, size_y={box.size_y})"
    )


# External function that takes a Box instance and returns the end coordinate as
# a numpy array
def get_box_end(box):
    return np.array(
        [box.origin_x + box.size_x, box.origin_y + box.size_y],
        dtype=np.float32,
    )


# Define the Box class
@jitclass(box_spec)
class Box:

    def __init__(self, origin_x, origin_y, size_x, size_y):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.size_x = size_x
        self.size_y = size_y

    def is_inside(self, x, y):
        return (
            self.origin_x <= x <= self.origin_x + self.size_x
            and self.origin_y <= y <= self.origin_y + self.size_y
        )
