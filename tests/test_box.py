import pytest
from dfsph.box import Box, box_repr


def test_is_inside():
    # Create a box with origin (0,0) and size (10, 10)
    box = Box(0.0, 0.0, 10.0, 10.0)

    # Points clearly inside the box.
    assert box.is_inside(5.0, 5.0) is True
    assert box.is_inside(0.0, 0.0) is True
    assert (
        box.is_inside(10.0, 10.0) is True
    )  # Assuming boundaries are inclusive

    # Points outside the box.
    assert box.is_inside(-1.0, 5.0) is False
    assert box.is_inside(5.0, -1.0) is False
    assert box.is_inside(11.0, 5.0) is False
    assert box.is_inside(5.0, 11.0) is False
