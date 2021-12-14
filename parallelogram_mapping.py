import logging
import argparse
import os
import cv2
import numpy as np
import transformations
import math
import random
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

def main(

):
    logging.info("parallelogram_mapping.main()")

    quad1 = GenerateQuadrilateralCorners()
    quad2 = GenerateQuadrilateralCorners()
    correspondences_dict = dict(zip(quad1, quad2))
    perspective_T = transformations.Perspective(correspondences_dict)
    persp_T_mtx = perspective_T.transformation_mtx
    logging.info("For general quadrilaterals, persp_T_mtx = {}".format(persp_T_mtx))
    ScatterPlot(quad1, quad2)

    paral1 = GenerateParallelogramCorners()
    paral2 = GenerateParallelogramCorners()
    correspondences_dict = dict(zip(paral1, paral2))
    perspective_T = transformations.Perspective(correspondences_dict)
    persp_T_mtx = perspective_T.transformation_mtx
    logging.info("For parallelograms, persp_T_mtx = {}".format(persp_T_mtx))
    ScatterPlot(paral1, paral2)



def GenerateQuadrilateralCorners():
    limit = 2000.0
    p1 = np.array([limit * np.random.random(), limit * np.random.random()])
    p2 = np.array([limit * np.random.random(), limit * np.random.random()])
    p3 = np.array([limit * np.random.random(), limit * np.random.random()])
    p1mp2 = p1 - p2
    p3mp1 = p3 - p1
    alpha = 0.1 + 0.9 * np.random.random()
    beta = 0.1 + 0.9 * np.random.random()
    p4 =  p1 + alpha * p1mp2 + beta * p3mp1
    return [tuple(p1), tuple(p2), tuple(p3), tuple(p4)]

def GenerateParallelogramCorners():
    limit = 2000.0
    p1 = np.array([limit * np.random.random(), limit * np.random.random()])
    p2 = np.array([limit * np.random.random(), limit * np.random.random()])
    p3 = np.array([limit * np.random.random(), limit * np.random.random()])
    p4 = p1 + p3 - p2
    return [tuple(p1), tuple(p2), tuple(p3), tuple(p4)]


def ScatterPlot(corners1, corners2):
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)]
    fig, ax = plt.subplots()

    line = np.arange(0, 1, 0.01)
    for corner_ndx in range(4):
        corner_1a = corners1[corner_ndx]
        if corner_ndx == 3:
            corner_1b = corners1[0]
        else:
            corner_1b = corners1[corner_ndx + 1]
        delta_x1 = corner_1b[0] - corner_1a[0]
        delta_y1 = corner_1b[1] - corner_1a[1]
        plt.plot(corner_1a[0] + delta_x1 * line, corner_1a[1] + delta_y1 * line, color='black')
        corner_2a = corners2[corner_ndx]
        if corner_ndx == 3:
            corner_2b = corners2[0]
        else:
            corner_2b = corners2[corner_ndx + 1]
        delta_x2 = corner_2b[0] - corner_2a[0]
        delta_y2 = corner_2b[1] - corner_2a[1]
        plt.plot(corner_2a[0] + delta_x2 * line, corner_2a[1] + delta_y2 * line, color='black')
    ax.scatter([c[0] for c in corners1], [c[1] for c in corners1], c=colors)
    ax.scatter([c[0] for c in corners2], [c[1] for c in corners2], c=colors, marker='^')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(

    )