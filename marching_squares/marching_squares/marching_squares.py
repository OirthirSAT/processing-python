import matplotlib.pyplot as plt  # This will import plotting module
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Any, Optional, cast


# typedefs
_NUMERIC_ARRAY = NDArray[np.floating[Any] | np.integer[Any]]
_POINT = tuple[int, int]
_VECTOR = tuple[_POINT, _POINT]


class MarchingSquares:

    @staticmethod
    def _readfile(
        filename: str | None, file: _NUMERIC_ARRAY | None, downsample_factor: float
    ) -> _NUMERIC_ARRAY:
        """Converts an input image to .hsv and performs downsampling

        Params:
            filename: the location at which the input image is stored. If file is None,
                this must not be None.
            file: the input image. If this is not None, it will be accepted regardless
                of the value of filename.
            downsample_factor: factor by which to scale the image. e.g. 0.5 corresponds
                to halving each dimension.

        Returns: The downsampled image in HSV format as a numeric numpy array.

        Throws:
            ValueError: If both filename and file are None.
        """

        if filename is None and file is None:
            raise ValueError("Expected one of filename or file to be not None.")

        if file is None:

            filename = cast(str, filename)  # MyPy needs some help

            image_bgr = cv2.imread(filename)

        else:

            file = cast(_NUMERIC_ARRAY, file)  # MyPy needs some help

            image_bgr = file[:, :, ::-1]

        # If necessary for performance speed, compress the file
        new_size: _POINT = (
            int(image_bgr.shape[0] * downsample_factor),
            int(image_bgr.shape[1] * downsample_factor),
        )
        image_resized = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)

        # For the chosen segmentation method it has been decided to segment
        # the image using the hue channel of a converted hsv image to
        # distinguish between land and sea.
        return cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    @staticmethod
    def _otsu_segmentation(image: _NUMERIC_ARRAY) -> tuple[float, _NUMERIC_ARRAY]:
        """Uses the Otsu segmentation method to distinguish between land and sea to
        extract the coastline vector. This will be later replaced by the UNET section
        of the pipeline. The Otsu threshold works by creating a histogram of the hue
        values in the hsv image. This will result in two large broad peaks in the
        histogram corresponding to the hue values of land more oranges and greens,
        whereas the sea will be distinctly blue. The threshold value is then the point
        between these two peaks. The output of this function is a binary valued
        segmented image 0 for sea and 1 for land
        """
        hue_channel = image[:, :, 0]
        return cv2.threshold(
            hue_channel,
            0,
            1,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    @staticmethod
    def _sort_key(point: _POINT) -> int:
        """Creating a function that will weight the dictionary points such that they
        can be sorted from left to right starting at the bottom left.
        """
        return point[1] * 100 + point[0]

    @staticmethod
    def _point_array(image: _NUMERIC_ARRAY) -> tuple[dict[_POINT, bool], int, int]:
        """This extracts the points from the image and stores them in a dictionary with
        each point either corresponding to black or white. The coordinates are doubled
        and incresed by one as vector lines will be drawn halfway between these points.
        This can be changed to match the original resolution of the image, however
        vectors will then be made of floating point coordinates.
        """

        black_list: list[_POINT] = []
        white_list: list[_POINT] = []

        x = image.shape[0] - 1
        y = image.shape[1] - 1

        for i in range(x + 1):
            for j in range(y + 1):
                if image[i][j] == 1:
                    black_list.append((j, x - i))
                else:
                    white_list.append((j, x - i))

        black = np.array(black_list) * 2 + 1
        white = np.array(white_list) * 2 + 1

        xblack: list[int] = [point[0] for point in black]
        yblack: list[int] = [point[1] for point in black]

        xwhite: list[int] = [point[0] for point in white]
        ywhite: list[int] = [point[1] for point in white]

        state: dict[_POINT, bool] = {tuple(point): True for point in black}
        state.update({tuple(point): False for point in white})
        sorted_dict: list[tuple[_POINT, bool]] = sorted(
            state.items(), key=lambda x: MarchingSquares._sort_key(x[0])
        )

        return (dict(sorted_dict), max(xblack + xwhite), max(yblack + ywhite))

    @staticmethod
    def _get_value(state_dict: dict[_POINT, bool], i: int, j: int) -> int:
        """Splitting the point array space into squares 1 pixel wide. These squares
        have corners lying on either a black or white point. The square as a whole
        adopts a value through the marching squares method, for a square centred at
        (2,2) it is corners at A(1,1),B(3,1),C(1,3) and D(3,3). Associating each of
        these corners with a binary weighting value A:2^0, B:2^2, C:2^3, D:2^4 and then
        summing these values multiplied by either 0 or 1 dependeing on the state of the
        point they sit on 1 for white and 0 for black will produce a value from 0 to
        15. Each of these values coresponds to a line shape which will be used to
        create a coastline vector.
        """

        A = int(state_dict[(i, j)])
        B = int(state_dict[(i + 2, j)])
        C = int(state_dict[(i, j + 2)])
        D = int(state_dict[(i + 2, j + 2)])
        return A + B * 2 + C * 4 + D * 8

    @staticmethod
    def _generate_edges(i: int, j: int, index: int) -> list[_VECTOR] | None:
        """Generates the line associated with the index of the square. This is done by
        outputting a start and end point for a line. Indexes of 6 and 9 are special in
        that two lines are created.
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
    def _list_vectors(
        state_dict: dict[_POINT, bool], x_len: int, y_len: int
    ) -> list[list[_VECTOR]]:

        vectors: list[list[_VECTOR] | None] = []
        i: int
        j: int
        for j in range(1, y_len, 2):

            for i in range(1, x_len, 2):

                index: int = MarchingSquares._get_value(state_dict, i, j)

                if index == 6 or index == 9:

                    double_vec: list[_VECTOR] | None = MarchingSquares._generate_edges(
                        i, j, index
                    )
                    if double_vec:
                        vectors.append([double_vec[0]])
                        vectors.append([double_vec[1]])

                else:
                    vectors.append(MarchingSquares._generate_edges(i, j, index))

        return [x for x in vectors if x is not None]  # filtering None values

    @staticmethod
    def _vector_shapes(vectors: list[list[_VECTOR]]) -> list[list[_POINT]]:
        """The purpose of this funciton is to connect all adjacent vector lines to
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

                    # Check if the vector connects to the shape
                    if vec[0] == end_point:
                        # append to end point
                        end_point = vec[1]
                        shape.append(end_point)
                        vectors_to_remove.remove(idx)
                        matched = True
                        break
                    elif vec[1] == end_point:
                        # append to start point
                        end_point = vec[0]
                        shape.append(end_point)
                        vectors_to_remove.remove(idx)
                        matched = True
                        break
                    elif vec[0] == start_point:

                        start_point = vec[1]
                        shape.insert(
                            0, start_point
                        )  # Add to the beginning of the shape
                        vectors_to_remove.remove(idx)
                        matched = True
                        break
                    elif vec[1] == start_point:
                        # If the start of the shape matches a reversed vector prepend it
                        start_point = vec[0]
                        shape.insert(0, start_point)
                        vectors_to_remove.remove(idx)
                        matched = True
                        break

            shapes.append(shape)

        return sorted(shapes, key=lambda shape: len(shape), reverse=True)

    @staticmethod
    def _show_coastline(
        image: _NUMERIC_ARRAY, shapes: list[list[_POINT]], x_len: int, y_len: int
    ) -> list[_POINT]:
        """This is the plotting function that will plot the main coastline vector. The
        range of the for loop can be changed to plot any islands as well.
        """

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Coastline Vector Extracted")
        for i in range(1):
            coastline_vector: list[_POINT] = shapes[i]
            xcoords: list[int] = []
            ycoords: list[int] = []
            for point in coastline_vector:
                xcoords.append(point[0])
                ycoords.append(point[1])
            plt.plot(xcoords, ycoords, linewidth=1)
        plt.xlim((0, x_len))
        plt.ylim((0, y_len))

        plt.subplot(1, 2, 2)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.axis("off")

        plt.show()
        return shapes[0]

    @staticmethod
    def run(
        filename: str | None, file: _NUMERIC_ARRAY | None, downsample_factor: float = 1
    ) -> None:

        image = MarchingSquares._readfile(filename, file, downsample_factor)
        _, threshold_image = MarchingSquares._otsu_segmentation(image)
        state_dict, x_len, y_len = MarchingSquares._point_array(threshold_image)
        vectors = MarchingSquares._list_vectors(state_dict, x_len, y_len)
        shapes = MarchingSquares._vector_shapes(vectors)
        _ = MarchingSquares._show_coastline(image, shapes, x_len, y_len)
