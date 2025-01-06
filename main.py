from cloud_mask import CloudMask
from unet import UNET_prediction
from marching_squares.marching_squares import MarchingSquares


def main(
    model_path: str,
    source_path: str,
    target_path: str,
    cloud_coverage_cutoff: float = 1.0,
):
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
    masked_image = cm.apply_cloud_mask(cm.create_cloud_mask())
    CloudMask.visualise_image(masked_image)


main(model_path="", source_path="Aberdeenshire_S2_20220810_TCI.tif", target_path="")
