import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Any
from cloud_mask.cloud_mask import CloudMask
# typedefs
_NUMERIC_ARRAY = NDArray[np.floating[Any] | np.integer[Any]]
_POINT = tuple[int, int]
_VECTOR = tuple[_POINT, _POINT]


class CoastlineExtractor_MS_altseg:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def plot_hue_histogram(ndwi: np.ndarray, value_range: tuple[int,int] = (0,255), bins: int = 200 ) -> None:
        """
        Plot the histogram of hue values that Otsu uses to determine the threshold.
        
        valid_hue: 1D array of 
        """
        # Ensure valid format
        values = ndwi.flatten()

        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=bins, range = value_range, edgecolor='black')
        plt.title("Histogram of NDWI values for Otsu Thresholding")
        plt.xlabel(f"NDWI Value ({value_range[0]}–{value_range[1]})")
        plt.ylabel("Pixel Count")
        plt.grid(alpha=0.3)
        plt.show()

    @staticmethod
    def _preprocess_image(masked_image: _NUMERIC_ARRAY, downsample_factor: float) -> tuple[_NUMERIC_ARRAY,_NUMERIC_ARRAY]:
        """
        Reads a tif file with bgr formatting, resizes the image and applies appropriate
        masking to NaN values and drops border artifacts.

        Args:
            masked_image: array
                The cloud masked image
            downsample_factor: float 
                Scaling factor for the image on each axis. e.g. 0.5
                applied to a 1024x1024 image results in a 512x512 image for a 4x
                reduction in pixels.

        Returns:
            resized_image : array
                Downsampled image with NaNs preserved.
            valid_mask : array (bool)
                Mask marking valid pixels after resizing.
        """
        
        print("[INFO] Preprocessing image for segmentation")
        
        #print(f"Number of nan values before downsampling {np.isnan(masked_image).sum()}")

        img = masked_image


        # Convert image to (H,W,bands) format
        if img.ndim == 2:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        elif img.ndim == 3 and img.shape[0] in (3, 4):
            img = np.moveaxis(img, 0, -1)


        img = img.astype(float)

        # Mask where all channels are finite.
        valid_mask_orig = np.isfinite(img).all(axis=-1)

        # If necessary for performance speed, compress the file.
        new_size: _POINT = (
            int(img.shape[1] * downsample_factor),
            int(img.shape[0] * downsample_factor),
        )

        # Use nearest neighbour interpolation, to avoid interpolating NaNs.
        img_resized: _NUMERIC_ARRAY = cv2.resize(
            img, new_size, interpolation=cv2.INTER_NEAREST
        )
        mask_resized: _NUMERIC_ARRAY = cv2.resize(
            valid_mask_orig.astype(np.uint8), new_size, interpolation = cv2.INTER_NEAREST
            ).astype(bool)

        #print(f"Number of NaN values after downsampling {np.isnan(img_resized).sum()}")

        # Remove border areas.
        border_mask = np.all(img_resized < 5, axis=-1)
        mask_resized[border_mask] = False

        # Apply mask.
        img_resized[~mask_resized] = np.nan

        #CloudMask.visualise_image(img_resized[:,:,:3]) - optional visualisation
        
        return img_resized , mask_resized

    @staticmethod
    def _otsu_segmentation_4channel(preprocessed_image: _NUMERIC_ARRAY, valid_mask: _NUMERIC_ARRAY) -> tuple[float, _NUMERIC_ARRAY]:
        """Use OTSU segmentation to classify land and sea.

        Uses the Otsu segmentation method to distinguish between land and sea to
        extract the coastline vector. The Otsu threshold works by creating a histogram of the NDWI
        values calculated using the formula NDWI = (G-NIR)/(G+NIR). This was chosen as through iterative testing
        it proved to be the most successful simple method to produce two large peaks with a noticable minimum
        which identifies a key threshold value to distinguish land and sea.

        Args:
            image: The image to be segmented.
            valid_mask: Mask of all NaN values

        Returns:
            ndwi_threshold: float
                The threshold value used to segment the image
            segmented_iamge: array
                Binary mask (0 = sea, 1 = land), Nans replaced with 16.
            threshold_valid: array
                OpenCVs raw thresholding output.

        """

        print("[INFO] Performing Otsu segmentation")
        
        image = preprocessed_image.copy()


        # Extract channels 
        green = image[:,:,1].astype(float)
        nir = image[:,:,3].astype(float)
        

        # Safely perform NDWI calculation
        denominator = green + nir
        eps = 1e-8
        denominator_safe = np.where(np.abs(denominator) < eps, np.nan, denominator)
        
        ndwi = (green-nir)/denominator_safe

        valid_ndwi = ndwi[valid_mask]
        valid_ndwi = valid_ndwi[~np.isnan(valid_ndwi)]


        #CoastlineExtractor_MS_altseg.plot_hue_histogram(valid_ndwi,(-1,1))


        # Perform Otsu segmentation on uint16 scaled NDWI
        scaled16 = ((valid_ndwi + 1.0) / 2.0 * 65535.0).clip(0,65535.0).astype(np.uint16)
        
        #CoastlineExtractor_MS_altseg.plot_hue_histogram(scaled16, range = (0,65535))

        img_for_thresh = scaled16.reshape(-1,1)

        threshold_value, threshold_valid = cv2.threshold(
            img_for_thresh, 
            0, 
            65535, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )


        # Convert back to original NDWI range (float) to apply thresholding
        ndwi_thresh = (threshold_value / 65535.0) * 2.0 - 1.0
        print(f"[STATS] NDWI threshold value: {ndwi_thresh}")

        threshold_full = np.full(ndwi.shape, np.nan, dtype = np.float32)
        nonan_mask = valid_mask & ~np.isnan(ndwi)
        threshold_full[valid_mask] = (ndwi[nonan_mask]> ndwi_thresh).astype(np.float32)
        
        
        # Create a masked array to hide NaNs
        masked_image = np.ma.masked_where(np.isnan(threshold_full), threshold_full)

        # Display using a custom colormap
        cmap = plt.cm.gray  
        cmap.set_bad(color='red')  # NaNs will show as red

        plt.imshow(masked_image, cmap=cmap)
        plt.colorbar(label='Segmentation Value')
        plt.title("Thresholded Image with NaNs")
        plt.show()

        #set the nan values to 16 such that they will fail any edge generation criteria in Marching Squares. 
        #This ensures regions surrounding masked pixels are vector free.

        segmented_image = np.nan_to_num(threshold_full, nan = 16)
       
        return ndwi_thresh, segmented_image, threshold_valid
    
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
        else:
            return None
        
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
    def upscale_and_shift_cv(img):
        """Used to shift the pixels of the original image to the scale used to generate the edges that will
        form coastline vectors."""

        upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        shifted = np.pad(upscaled, ((1, 0), (1, 0), (0, 0)), mode='constant')
        return shifted

    @staticmethod
    def _create_overlay(img, valid_mask, shapes):
        """
        Create a coastline overlay on top of the preprocessed image.
        Returns an RGB float image in the range [0,1].
        """

        # Normalise the image to range [0,1]
        rgb = img[:,:,:3].astype(np.float32)
        
        vmin = np.nanmin(rgb)
        vmax = np.nanmax(rgb)

        if vmax <= 1.0 and vmin >= 0.0:
        # Float is already consistent
            norm = rgb
        elif vmax > 1.0 and vmax <= 255.0 and vmin >= 0.0:
            # uint8 conversion
            norm = rgb / 255.0
        else:
            # Other cases
            if np.isclose(vmax, vmin):
                norm = np.clip(rgb - vmin, 0.0, 1.0)  # constant image
            else:
                norm = (rgb - vmin) / (vmax - vmin)
        

        # Apply our valid_mask across all channels
        mask3d = np.stack([valid_mask, valid_mask, valid_mask], axis = -1)
        norm[~mask3d] = 0.0


        # Upscale and shift into our new coordinate space (used for Marching Squares)
        shifted = CoastlineExtractor_MS_altseg.upscale_and_shift_cv(norm)
        shifted_uint8 = (shifted * 255).astype(np.uint8).copy()
        H,W = shifted.shape[:2]

        # Prepare rgba image for drawing lines
        overlay_rgba = np.dstack((shifted_uint8, np.full((H,W), 255, dtype = np.uint8)))
        overlay_rgba = np.ascontiguousarray(overlay_rgba)
        overlay_bgr = np.ascontiguousarray(overlay_rgba[:, :, :3])


        # Draw Lines
        # Flip the coordinates of our shapes array such that they correctly overlay on top of our image
        shapes_transformed = [[[x, W - y] for x, y in line] for line in shapes]

        for line in shapes_transformed[:10]:
            pts = np.array(line, dtype = np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                overlay_bgr, 
                [pts], 
                isClosed=False, 
                color=(255, 255, 255), 
                thickness=1,
                lineType=cv2.LINE_AA 
                )
        overlay_rgba[:, :, :3] = overlay_bgr


        #Use alpha channel to blen the overlay with the original shifted image
        alpha = 0.6
        rgb_overlay = overlay_rgba[:, :, :3].astype(np.float32) / 255.0
        alpha_map = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0) * alpha

        out = (alpha_map[..., None] * rgb_overlay) + ((1.0 - alpha_map[..., None]) * shifted)
        output = np.clip(out, 0, 1)

        return output
    
    @staticmethod
    def run(masked_image: _NUMERIC_ARRAY, downsample_factor: float = 1) -> dict[str, Any]:
        preprocessed_image, valid_mask = CoastlineExtractor_MS_altseg._preprocess_image(masked_image, downsample_factor)
        _, threshold_image, threshold_value = CoastlineExtractor_MS_altseg._otsu_segmentation_4channel(preprocessed_image, valid_mask)
        print("[INFO] Running Marching Squares")
        state_array, x_len, y_len = CoastlineExtractor_MS_altseg._point_array(threshold_image)
        vectors = CoastlineExtractor_MS_altseg._list_vectors(state_array, x_len, y_len)
        shapes = CoastlineExtractor_MS_altseg._vector_shapes(vectors)
        
        print(f"[STATS] Length of the coastline vector: {len(shapes[0])}")
        
        img_overlay = CoastlineExtractor_MS_altseg._create_overlay(preprocessed_image,valid_mask,shapes)
        
        return {
            "preprocessed_image" : preprocessed_image, 
            "shapes": shapes,
            "vectors": vectors,
            "threshold_image": threshold_image,
            "overlay_image": img_overlay
        }
