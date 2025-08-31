import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Any

# typedefs
_NUMERIC_ARRAY = NDArray[np.floating[Any] | np.integer[Any]]
_POINT = tuple[int, int]
_VECTOR = tuple[_POINT, _POINT]

class MarchingSquaresRefiner:
    """
    Takes a binary coastline mask (from Otsu, NDWI, etc.)
    and generates sub-pixel accurate vectors using Marching Squares.
    """

    @staticmethod
    def _point_array(image: _NUMERIC_ARRAY) -> tuple[NDArray[np.bool_], int, int]:
        state_array: NDArray[np.bool_] = image[::-1, :]
        y_len, x_len = np.array(image.shape) * 2 - 1
        return (state_array, y_len, x_len)

    @staticmethod
    def _get_value(state_array: NDArray[np.bool_], i: int, j: int) -> int:
        _i = (i - 1) // 2
        _j = (j - 1) // 2

        # Corners
        A = int(state_array[_j, _i])
        B = int(state_array[_j, _i + 1])
        C = int(state_array[_j + 1, _i])
        D = int(state_array[_j + 1, _i + 1])

        return A + B * 2 + C * 4 + D * 8

    @staticmethod
    def _generate_edges(i: int, j: int, index: int) -> list[_VECTOR] | None:
        x, y = i, j
        vector: list[_VECTOR] = []
        if index == 0 or index == 15:
            return None
        elif index in (1, 14):
            vector.append(((x + 1, y), (x, y + 1)))
        elif index in (2, 13):
            vector.append(((x + 1, y), (x + 2, y + 1)))
        elif index in (3, 12):
            vector.append(((x, y + 1), (x + 2, y + 1)))
        elif index in (7, 8):
            vector.append(((x + 2, y + 1), (x + 1, y + 2)))
        elif index == 9:
            vector.append(((x, y + 1), (x + 1, y + 2)))
            vector.append(((x + 1, y), (x + 2, y + 1)))
        elif index in (5, 10):
            vector.append(((x + 1, y), (x + 1, y + 2)))
        elif index in (4, 11):
            vector.append(((x, y + 1), (x + 1, y + 2)))
        elif index == 6:
            vector.append(((x + 2, y + 1), (x + 1, y + 2)))
            vector.append(((x + 1, y), (x, y + 1)))
        return vector

    @staticmethod
    def _list_vectors(state_array: NDArray[np.bool_], x_len: int, y_len: int) -> list[list[_VECTOR]]:
        vectors: list[list[_VECTOR] | None] = []
        for j in range(1, y_len, 2):
            for i in range(1, x_len, 2):
                index = MarchingSquaresRefiner._get_value(state_array, i, j)
                if index in (6, 9):
                    double_vec = MarchingSquaresRefiner._generate_edges(i, j, index)
                    if double_vec:
                        vectors.append([double_vec[0]])
                        vectors.append([double_vec[1]])
                else:
                    vectors.append(MarchingSquaresRefiner._generate_edges(i, j, index))
        return [x for x in vectors if x is not None]

    @staticmethod
    def _vector_shapes(vectors: list[list[_VECTOR]]) -> list[list[_POINT]]:
        shapes: list[list[_POINT]] = []
        vectors_to_remove: set[int] = set(range(len(vectors)))
        while vectors_to_remove:
            shape: list[_POINT] = []
            vector = vectors[vectors_to_remove.pop()][0]
            start_point, end_point = vector
            shape.extend([start_point, end_point])
            matched = True
            while matched:
                matched = False
                for idx in list(vectors_to_remove):
                    vec = vectors[idx][0]
                    if vec[0] == end_point:
                        end_point = vec[1]
                        shape.append(end_point)
                        vectors_to_remove.remove(idx)
                        matched = True
                        break
                    elif vec[1] == end_point:
                        end_point = vec[0]
                        shape.append(end_point)
                        vectors_to_remove.remove(idx)
                        matched = True
                        break
                    elif vec[0] == start_point:
                        start_point = vec[1]
                        shape.insert(0, start_point)
                        vectors_to_remove.remove(idx)
                        matched = True
                        break
                    elif vec[1] == start_point:
                        start_point = vec[0]
                        shape.insert(0, start_point)
                        vectors_to_remove.remove(idx)
                        matched = True
                        break
            shapes.append(shape)
        return sorted(shapes, key=lambda s: len(s), reverse=True)

    @staticmethod
    def run(binary_mask: _NUMERIC_ARRAY, debug: bool = False) -> list[_POINT]:
        """Run marching squares refinement on a binary mask"""
        state_array, x_len, y_len = MarchingSquaresRefiner._point_array(binary_mask.astype(bool))
        vectors = MarchingSquaresRefiner._list_vectors(state_array, x_len, y_len)
        shapes = MarchingSquaresRefiner._vector_shapes(vectors)
        if debug and shapes:
            plt.figure(figsize=(8, 6))
            coastline_vector = shapes[0]
            xs, ys = zip(*coastline_vector)
            plt.plot(xs, ys, linewidth=1)
            plt.title("Marching Squares Sub-Pixel Coastline")
            plt.show()
        return shapes[0] if shapes else []
