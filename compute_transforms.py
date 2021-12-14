import logging
import argparse
import os
import cv2
import numpy as np
import transformations

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

    feature_points1 = np.array([[1295, 538], [3226, 626], [3637, 2724], [1637, 2982]], dtype=np.float32)
    feature_points2 = np.array([[1365, 562], [3451, 532], [3771, 2893], [1675, 2956]], dtype=np.float32)
    warped_feature_points = np.array([[100, 100], [1100, 100], [1100, 1100], [100, 1100]], dtype=np.float32)

    # Draw the location of A, B, C, D
    DrawABCD(image1, feature_points1)
    DrawABCD(image2, feature_points2)
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
    CircleFixedPoints(warped_affine_img1, warped_feature_points)
    CircleFixedPoints(warped_affine_img2, warped_feature_points)

    warped_affine_img1_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedAffine1.png")
    cv2.imwrite(warped_affine_img1_filepath, warped_affine_img1)
    warped_affine_img2_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedAffine2.png")
    cv2.imwrite(warped_affine_img2_filepath, warped_affine_img2)
    window_original = cv2.namedWindow("Affine warped image", cv2.WINDOW_NORMAL)
    cv2.imshow("Affine warped image", warped_affine_img1)
    cv2.waitKey(0)
    cv2.imshow("Affine warped image", warped_affine_img2)
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
    CircleFixedPoints(warped_perspective_img1, warped_feature_points)
    CircleFixedPoints(warped_perspective_img2, warped_feature_points)

    warped_perspective_img1_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedPerspective1.png")
    cv2.imwrite(warped_perspective_img1_filepath, warped_perspective_img1)
    warped_perspective_img2_filepath = os.path.join(outputDirectory, "computeTransforms_main_warpedPerspective2.png")
    cv2.imwrite(warped_perspective_img2_filepath, warped_perspective_img2)

    window_warped = cv2.namedWindow("Perspective warped image", cv2.WINDOW_NORMAL)
    cv2.imshow("Perspective warped image", warped_perspective_img1)
    cv2.waitKey(0)
    cv2.imshow("Perspective warped image", warped_perspective_img2)
    cv2.waitKey(0)

    # Compute the perspective transformation by solving a homogeneous system of linear equations
    correspondences1 = {}
    for pt_ndx in range(feature_points1.shape[0]):
        correspondences1[tuple(warped_feature_points[pt_ndx].tolist())] = tuple(feature_points1[pt_ndx].tolist())
    perspective_T1 = transformations.Perspective(correspondences1)
    logging.info("perspective_T1.transformation_mtx =\n{}".format(perspective_T1.transformation_mtx))
    correspondences2 = {}
    for pt_ndx in range(feature_points2.shape[0]):
        correspondences2[tuple(warped_feature_points[pt_ndx].tolist())] = tuple(feature_points2[pt_ndx].tolist())
    perspective_T2 = transformations.Perspective(correspondences2)
    logging.info("perspective_T2.transformation_mtx =\n{}".format(perspective_T2.transformation_mtx))

def DrawABCD(image, points_arr):
    ABCD = ['A', 'B', 'C', 'D']
    for point_ndx in range(points_arr.shape[0]):
        point = points_arr[point_ndx]
        point = (round(point[0]), round(point[1]))
        cv2.circle(image, point, 13, (255, 0, 0), thickness=-1)
        cv2.putText(image, ABCD[point_ndx], (point[0] - 40, point[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0),
                    thickness=6)

def CircleFixedPoints(image, fixed_points_arr):
    for point_ndx in range(fixed_points_arr.shape[0]):
        point = fixed_points_arr[point_ndx]
        point = (round(point[0]), round(point[1]))
        cv2.circle(image, point, 31, (0, 0, 255), thickness=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFilepath1', help="The filepath to the 1st image. Default: './images/board_1m.jpg'", default='./images/board_1m.jpg')
    parser.add_argument('--imageFilepath2', help="The filepath to the 2nd image. Default: './images/board_7m.jpg'", default='./images/board_7m.jpg')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './compute_transforms_outputs'",
                        default='./compute_transforms_outputs')
    args = parser.parse_args()
    main(args.imageFilepath1,
         args.imageFilepath2,
         args.outputDirectory)