import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Any

# typedefs
_NUMERIC_ARRAY = NDArray[np.floating[Any] | np.integer[Any]]
_POINT = tuple[int, int]
_VECTOR = tuple[_POINT, _POINT]


class CoastlineExtractor_MS_altseg:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    @staticmethod
    def _preprocess_image(masked_image: _NUMERIC_ARRAY, downsample_factor: float) -> _NUMERIC_ARRAY:
        if masked_image.ndim == 2:
            masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif masked_image.shape[0] in (3, 4):
            masked_image = np.moveaxis(masked_image, 0, -1)

        new_size: _POINT = (
            int(masked_image.shape[1] * downsample_factor),
            int(masked_image.shape[0] * downsample_factor),
        )
        image_resized = cv2.resize(masked_image, new_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    @staticmethod
    def _otsu_segmentation(image: _NUMERIC_ARRAY) -> tuple[float, _NUMERIC_ARRAY]:
        hue_channel = image[:, :, 0]
        return cv2.threshold(hue_channel, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    @staticmethod
    def _point_array(image: _NUMERIC_ARRAY) -> tuple[NDArray[np.bool], int, int]:
        state_array: NDArray[np.bool] = image[::-1, :]
        y_len, x_len = np.array(image.shape) * 2 - 1
        return state_array, y_len, x_len

    @staticmethod
    def _get_value(state_array: NDArray[np.bool], i: int, j: int) -> int:
        _i = (i - 1) // 2
        _j = (j - 1) // 2
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
    def _list_vectors(state_array: NDArray[np.bool], x_len: int, y_len: int) -> list[list[_VECTOR]]:
        vectors: list[list[_VECTOR] | None] = []
        for j in range(1, y_len, 2):
            for i in range(1, x_len, 2):
                index = CoastlineExtractor_MS_altseg._get_value(state_array, i, j)
                if index in (6, 9):
                    double_vec = CoastlineExtractor_MS_altseg._generate_edges(i, j, index)
                    if double_vec:
                        vectors.append([double_vec[0]])
                        vectors.append([double_vec[1]])
                else:
                    vectors.append(CoastlineExtractor_MS_altseg._generate_edges(i, j, index))
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
                        end_point = vec[1]; shape.append(end_point)
                        vectors_to_remove.remove(idx); matched = True; break
                    elif vec[1] == end_point:
                        end_point = vec[0]; shape.append(end_point)
                        vectors_to_remove.remove(idx); matched = True; break
                    elif vec[0] == start_point:
                        start_point = vec[1]; shape.insert(0, start_point)
                        vectors_to_remove.remove(idx); matched = True; break
                    elif vec[1] == start_point:
                        start_point = vec[0]; shape.insert(0, start_point)
                        vectors_to_remove.remove(idx); matched = True; break
            shapes.append(shape)

        return sorted(shapes, key=lambda shape: len(shape), reverse=True)

    @staticmethod
    def run(masked_image: _NUMERIC_ARRAY, downsample_factor: float = 1) -> dict[str, Any]:
        hsv_image = CoastlineExtractor_MS_altseg._preprocess_image(masked_image, downsample_factor)
        _, threshold_image = CoastlineExtractor_MS_altseg._otsu_segmentation(hsv_image)
        state_array, x_len, y_len = CoastlineExtractor_MS_altseg._point_array(threshold_image)
        vectors = CoastlineExtractor_MS_altseg._list_vectors(state_array, x_len, y_len)
        shapes = CoastlineExtractor_MS_altseg._vector_shapes(vectors)

        overlay = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        for shape in shapes[:1]:
            pts = np.array(shape, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

        return {
            "shapes": shapes,
            "vectors": vectors,
            "threshold_image": threshold_image,
            "overlay_image": overlay
        }
