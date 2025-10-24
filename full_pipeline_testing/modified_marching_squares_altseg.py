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
        """
        Reads a tif file with bgr formatting, resizes the image if necessary and
        then converts into a hsv file using the cv2 library.

        Args:
            masked_image: The cloud masked image
            downsample_factor: float by which to scale the image on each axis. e.g. 0.5
                applied to a 1024x1024 image results in a 512x512 image for a 4x
                reduction in pixels.

        Returns:
            A numeric array representing the image after being downscaled and converted
            to hsv format.
        """
        print(f"Number of nan values before downsampling {np.isnan(masked_image).sum()}")

        if masked_image.ndim == 2:
            masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif masked_image.shape[0] in (3, 4):
            masked_image = np.moveaxis(masked_image, 0, -1)

        # If necessary for performance speed, compress the file
        new_size: _POINT = (
            int(masked_image.shape[1] * downsample_factor),
            int(masked_image.shape[0] * downsample_factor),
        )
        image_resized: _NUMERIC_ARRAY = cv2.resize(
            masked_image, new_size, interpolation=cv2.INTER_AREA
        )
        print(f"Number of nan values after downsampling {np.isnan(image_resized).sum()}")
        valid_mask = ~np.isnan(image_resized).any(axis = -1)
        image_filled = np.nan_to_num(image_resized, nan = 0.0)
        image_resized_uint8= image_filled.astype(np.uint8)

        # For the chosen segmentation method it has been decided to segment
        # the image using the hue channel of a converted hsv image to
        # distinguish between land and sea.
        print("dtype before conversion:", image_resized_uint8.dtype)

        return cv2.cvtColor(image_resized_uint8, cv2.COLOR_BGR2HSV) , valid_mask

    @staticmethod
    def _otsu_segmentation(image: _NUMERIC_ARRAY, valid_mask: _NUMERIC_ARRAY) -> tuple[float, _NUMERIC_ARRAY]:
        """Use OTSU segmentation to classify land and sea.

        Uses the Otsu segmentation method to distinguish between land and sea to
        extract the coastline vector. This will be later replaced by the UNET section
        of the pipeline. The Otsu threshold works by creating a histogram of the hue
        values in the hsv image. This will result in two large broad peaks in the
        histogram corresponding to the hue values of land more oranges and greens,
        whereas the sea will be distinctly blue. The threshold value is then the point
        between these two peaks.

        Args:
            image: The image to be segmented in hsv format.

        Returns:
            A tuple containing [0] the threshold value between land and sea and [1] a
            binary valued segmented image where 0 represents sea and 1 land.
        """

        hue: _NUMERIC_ARRAY = image[:, :, 0]
        valid_hue = hue[valid_mask]
        
        threshold_value, threshold_valid = cv2.threshold(
            valid_hue, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        print(f"Shape of threshold_valid matrix: {threshold_valid.shape}")
        print(f"Shape of valid_hue matrix: {valid_hue.shape}")
        threshold_full = np.full(hue.shape, np.nan, dtype = np.float32)
        threshold_full[valid_mask] = (hue[valid_mask]> threshold_value).astype(np.float32)
        print(f"Shape of threshold matrix: {threshold_full.shape}")
        print(f"threshold value: {threshold_value}")
        print(f"max value of hue array: {hue.max()}")
        print(f"occurences of nan {np.isnan(threshold_full).sum()}")
        # Create a masked array to hide NaNs
        masked_image = np.ma.masked_where(np.isnan(threshold_full), threshold_full)

        # Display using a custom colormap
        cmap = plt.cm.gray  # or use 'viridis', 'plasma', etc.
        cmap.set_bad(color='red')  # NaNs will show as red

        plt.imshow(masked_image, cmap=cmap)
        plt.colorbar(label='Segmentation Value')
        plt.title("Thresholded Image with NaNs")
        plt.show()

        
        return threshold_value, threshold_full, threshold_valid
    
    @staticmethod
    def _point_array(image: _NUMERIC_ARRAY) -> tuple[NDArray[bool], int, int]:
        """Convert image points to lookup array.

        This extracts the points from the image and stores them in an array with each
        point either corresponding to black or white. The coordinates for x_len and
        y_len are doubled and decreased by one as vector lines will be drawn halfway
        between these points. This can be changed to match the original resolution of
        the image, however vectors will then be made of floating point coordinates.

        Args:
            image: An array representing the image in hsv.

        Returns:
            A tuple containing [0] an array of boolean values for each point, where
            a value of True designates a black point and False a white point; [1], [2]
            the new height and width of the image, expanded for marching squares.
        """

        state_array: NDArray[bool] = image[::-1, :]
        y_len, x_len = np.array(image.shape) * 2 - 1
        
        return state_array, y_len, x_len

    @staticmethod
    def _get_value(state_array: NDArray[bool], i: int, j: int) -> int:
        """Compute weighted marching squares pixel value.

        Splitting the point array space into squares 1 pixel wide. These squares
        have corners lying on either a black or white point. The square as a whole
        adopts a value through the marching squares method, for a square centred at
        (2,2) it is corners at A(1,1),B(3,1),C(1,3) and D(3,3). Associating each of
        these corners with a binary weighting value A:2^0, B:2^1, C:2^2, D:2^3 and then
        summing these values multiplied by either 0 or 1 depending on the state of the
        point they sit on 1 for white and 0 for black will produce a value from 0 to
        15. Each of these values corresponds to a line shape which will be used to
        create a coastline vector.

        Args:
            state_array: A mapping from point to a bool representing its colour, where
                True = black and False = white.
            i: The height value of the point to evaluate.
            j: The width value of the point to evaluate.

        Returns:
            An integer corresponding to the shape of the line that will be drawn
            through this point.
        """
        # Convert to state_array coordinates
        _i = (i - 1) // 2
        _j = (j - 1) // 2

        # Compute corner values
        A = int(state_array[_j, _i])
        B = int(state_array[_j, _i + 1])
        C = int(state_array[_j + 1, _i])
        D = int(state_array[_j + 1, _i + 1])

        return A + B * 2 + C * 4 + D * 8

    @staticmethod
    def _generate_edges(i: int, j: int, index: int) -> list[_VECTOR] | None:
        """Generate edge line associated with square index.

        Generates the line associated with the index of the square. This is done by
        outputting a start and end point for a line. Indexes of 6 and 9 are special in
        that two lines are created.

        Args:
            i: The height value of the point to evaluate.
            j: The width value of the point to evaluate.
            index: An integer representing the shape of the line to be drawn through
                point [i, j].

        Returns:
            A list of line segments representing the borders drawn by the marching
            squares algorithm.
        """

        x: int
        y: int
        x, y = i, j
        vector: list[_VECTOR] = []
        start: _POINT
        end: _POINT

        if index == 0 or index == 15:
            return None
        elif index == 1 or index == 14:
            start = (x + 1, y)
            end = (x, y + 1)
            vector.append((start, end))
        elif index == 2 or index == 13:
            start = (x + 1, y)
            end = (x + 2, y + 1)
            vector.append((start, end))
        elif index == 3 or index == 12:
            start = (x, y + 1)
            end = (x + 2, y + 1)
            vector.append((start, end))
        elif index == 7 or index == 8:
            start = (x + 2, y + 1)
            end = (x + 1, y + 2)
            vector.append((start, end))
        elif index == 9:
            start = (x, y + 1)
            end = (x + 1, y + 2)
            vector.append((start, end))
            start = (x + 1, y)
            end = (x + 2, y + 1)
            vector.append((start, end))
        elif index == 5 or index == 10:
            start = (x + 1, y)
            end = (x + 1, y + 2)
            vector.append((start, end))
        elif index == 4 or index == 11:
            start = (x, y + 1)
            end = (x + 1, y + 2)
            vector.append((start, end))
        elif index == 6:
            start = (x + 2, y + 1)
            end = (x + 1, y + 2)
            vector.append((start, end))
            start = (x + 1, y)
            end = (x, y + 1)
            vector.append((start, end))

        return vector

    @staticmethod
    def _list_vectors(state_array: NDArray[bool], x_len: int, y_len: int) -> list[list[_VECTOR]]:
        """
        Args:
            state_array: A mapping of points to their black/white state, with True
                representing a black state and False a white state.
            x_len: The width of the image represented by state_array.
            y_len: The height of the image represented by state_array.

        Returns:
            A list of lists of line segments representing the border generated by
            marching squares. Each inner list of line segments represents part of the
            border generated by a single pixel in the source image.
        """

        vectors: list[list[_VECTOR] | None] = []
        i: int
        j: int
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
        """Merge adjacent vector lines into coastline vector.

        The purpose of this funciton is to connect all adjacent vector lines to
        create one long "coastline vector". This is done by creating a set of the
        vector lines from the previous function. The first in this set is popped out
        and the start and end points are added to a shape vector. The set
        vectors_to_remove is then looped through until the start or end point of one of
        these vectors "matches" the start or end point of the popped vector. The
        matched vector is then added to the shape vector, for example if the start
        point of the popped vector matched the end point of the matched vector then the
        start coordinate of the matched vector will be added to the shape array. This
        is then repeated with the new start and end vector of the shape until there is
        no match. In this circumstance, the shape is appended to a "shapes" array and a
        new shape is created and the process repeats until there are no vectors left to
        remove. At the end of this function, the shapes are ordered dependeing on their
        size. The main coastline vector will be the longest whereas there will be
        shorter vectors corresponding to islands.

        Args:
            vectors: The list of line segment lists representing the contributions to
                the sea-land border by every pixel.

        Returns:
            A list of joined shapes, represented as lists of points, ordered by shape
            size (number of points in the shape).
        """
        shapes: list[list[_POINT]] = []
        vectors_to_remove: set[int] = set(range(len(vectors)))

        while vectors_to_remove:
            shape: list[_POINT] = []
            
            # Get the first vector and extract the tuple of points
            vector: _VECTOR = vectors[vectors_to_remove.pop()][0]
            
            start_point: _POINT
            end_point: _POINT

            start_point, end_point = vector

            # Add the start and end points to the shape
            shape.extend([start_point, end_point])
            matched: bool = True

            while matched:

                matched = False
                
                idx: int

                for idx in list(vectors_to_remove):
                    
                    vec: _VECTOR = vectors[idx][0]
                    
                    #check if the vector connects to the shape
                    if vec[0] == end_point:
                        # append to end point
                        end_point = vec[1]; shape.append(end_point)
                        vectors_to_remove.remove(idx); matched = True; break
                    elif vec[1] == end_point:
                        # append to start point
                        end_point = vec[0]; shape.append(end_point)
                        vectors_to_remove.remove(idx); matched = True; break
                    elif vec[0] == start_point:
                        # Add to the beginning of the shape
                        start_point = vec[1]; shape.insert(0, start_point)
                        vectors_to_remove.remove(idx); matched = True; break
                    elif vec[1] == start_point:
                        # If the start of the shape matches a reversed vector prepend it
                        start_point = vec[0]; shape.insert(0, start_point)
                        vectors_to_remove.remove(idx); matched = True; break
            
            shapes.append(shape)

        return sorted(shapes, key=lambda shape: len(shape), reverse=True)

    @staticmethod
    def run(masked_image: _NUMERIC_ARRAY, downsample_factor: float = 1) -> dict[str, Any]:
        hsv_image, valid_mask = CoastlineExtractor_MS_altseg._preprocess_image(masked_image, downsample_factor)
        _, threshold_image, threshold_value = CoastlineExtractor_MS_altseg._otsu_segmentation(hsv_image, valid_mask)
        state_array, x_len, y_len = CoastlineExtractor_MS_altseg._point_array(threshold_value)
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
