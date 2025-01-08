from cloud_mask import CloudMask
from unet import UNET_prediction
from marching_squares.marching_squares import MarchingSquares
import logging
import matplotlib.pyplot as plt
import numpy as np


def main(
    model_path: str,
    source_path: str,
    target_path: str,
    cloud_coverage_cutoff: float = 1.0,
) -> None:
    """
    Params:
        model_path: path to coastline detection tensorflow model.
        source_path: path to source image (to be processed). Only .tif files are
                supported.
        target_path: path to file where the vector coastline output should be saved.
        cloud_coverage_cutoff: The ratio of cloud to image above which the run should
                be aborted. Defaults to 1.0: all images will be accepted.
    """
    cm = CloudMask(tif_path=source_path)
    mask = cm.create_cloud_mask()
    cloud_cover_fraction = cm.cloud_cover_fraction(mask)
    if cloud_cover_fraction > cloud_coverage_cutoff:
        logging.warning(
            f"Calculated cloud coverage {cloud_cover_fraction} > provided cutoff "
            f"{cloud_coverage_cutoff} for {source_path}. Run aborted."
        )
        return
    masked_image = cm.apply_cloud_mask(mask)
    unet_prediction = (
        UNET_prediction.make_prediction(
            model_path, source_image=masked_image[:, :, :3]
        )[0]
        * 256
    ).astype(np.uint8)
    MarchingSquares.run(None, unet_prediction, 1)


main(
    model_path="unet/unet_coastline_model.h5",
    source_path="Aberdeenshire_S2_20220810_TCI.tif",
    target_path="",
)
