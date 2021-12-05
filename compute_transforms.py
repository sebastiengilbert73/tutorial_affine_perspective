import logging
import argparse
import os
import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main(
        imageFilepath1,
        imageFilepath2,
        outputDirectory
):
    logging.info("compute_transforms.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the image
    image1 = cv2.imread(imageFilepath1, cv2.IMREAD_COLOR)
    image2 = cv2.imread(imageFilepath2, cv2.IMREAD_COLOR)

    feature_points1 = np.array([[1365, 562], [3451, 532], [3771, 2893], [1675, 2956]], dtype=np.float32)
    feature_points2 = np.array([[1295, 538], [3226, 626], [3637, 2724], [1637, 2982]], dtype=np.float32)
    warped_feature_points = np.array([[100, 100], [1100, 100], [1100, 1100], [100, 1100]], dtype=np.float32)

    # Affine transform
    # With OpenCV
    affine_mtx1 = cv2.getAffineTransform(feature_points1[:3, :], warped_feature_points[:3, :])
    affine_mtx2 = cv2.getAffineTransform(feature_points2[:3, :], warped_feature_points[:3, :])
    logging.info("affine_mtx1 =\n{}".format(affine_mtx1))
    logging.info("affine_mtx2 =\n{}".format(affine_mtx2))
    # Warp affine
    warped_image_size = (1200, 1200)
    warped_affine_img1 = cv2.warpAffine(image1, affine_mtx1, dsize=warped_image_size)
    warped_affine_img2 = cv2.warpAffine(image2, affine_mtx2, dsize=warped_image_size)
    warped_affine_img1_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedAffine1.png")
    cv2.imwrite(warped_affine_img1_filepath, warped_affine_img1)
    warped_affine_img2_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedAffine2.png")
    cv2.imwrite(warped_affine_img2_filepath, warped_affine_img2)
    cv2.imshow("Affine warped 1", warped_affine_img1)
    cv2.waitKey(0)
    cv2.imshow("Affine warped 2", warped_affine_img2)
    cv2.waitKey(0)

    # Perspective transform
    # With OpenCV
    perspective_mtx1 = cv2.getPerspectiveTransform(feature_points1, warped_feature_points)
    perspective_mtx2 = cv2.getPerspectiveTransform(feature_points2, warped_feature_points)
    logging.info("perspective_mtx1 =\n{}".format(perspective_mtx1))
    logging.info("perspective_mtx2 =\n{}".format(perspective_mtx2))
    # Warp perspective
    warped_perspective_img1 = cv2.warpPerspective(image1, perspective_mtx1, dsize=warped_image_size)
    warped_perspective_img2 = cv2.warpPerspective(image2, perspective_mtx2, dsize=warped_image_size)
    warped_perspective_img1_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedPerspective1.png")
    cv2.imwrite(warped_perspective_img1_filepath, warped_perspective_img1)
    warped_perspective_img2_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedPerspective2.png")
    cv2.imwrite(warped_perspective_img2_filepath, warped_perspective_img2)
    cv2.imshow("Perspective warped 1", warped_perspective_img1)
    cv2.waitKey(0)
    cv2.imshow("Perspective warped 2", warped_perspective_img2)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFilepath1', help="The filepath to the 1st image. Default: './images/IMG_6660.JPG'", default='./images/IMG_6660.JPG')
    parser.add_argument('--imageFilepath2', help="The filepath to the 2nd image. Default: './images/IMG_6661.JPG'", default='./images/IMG_6661.JPG')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './compute_transforms_outputs'",
                        default='./compute_transforms_outputs')
    args = parser.parse_args()
    main(args.imageFilepath1,
         args.imageFilepath2,
         args.outputDirectory)