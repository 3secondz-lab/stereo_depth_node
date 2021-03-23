import os

import cv2
import numpy as np


# Stereo image preprocessor
class StereoImagePreprocessor:
    def __init__(self, path_calib_left, path_calib_right, width, height,
                 crop_top=0, crop_bottom=0, crop_left=0, crop_right=0):
        # Path to calibration files
        self.path_calib_left = path_calib_left
        self.path_calib_right = path_calib_right

        # Image size
        self.w = width
        self.h = height

        # Crop parameters
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right

        assert os.path.exists(self.path_calib_left), \
            "File does not exist: {}".format(self.path_calib_left)
        assert os.path.exists(self.path_calib_right), \
            "File does not exist: {}".format(self.path_calib_right)

        assert self.w > 0 and self.h > 0
        assert self.crop_top >= 0 and self.crop_bottom >= 0 \
               and self.crop_left >= 0 and self.crop_right >= 0

        # Read calibration files
        calib_left = cv2.FileStorage(self.path_calib_left,
                                     cv2.FILE_STORAGE_READ)
        K1 = calib_left.getNode("K").mat()
        D1 = calib_left.getNode("D").mat()
        # R1 = calib_left.getNode("R").mat()
        # T1 = calib_left.getNode("T").mat() / 1000.0

        calib_left.release()

        calib_right = cv2.FileStorage(self.path_calib_right,
                                      cv2.FILE_STORAGE_READ)
        K2 = calib_right.getNode("K").mat()
        D2 = calib_right.getNode("D").mat()
        R2 = calib_right.getNode("R").mat()
        T2 = calib_right.getNode("T").mat() / 1000.0

        calib_right.release()

        if self.crop_left > 0:
            K1[0, 2] = K1[0, 2] - self.crop_left
            K2[0, 2] = K2[0, 2] - self.crop_left
            self.w = self.w - self.crop_left
        if self.crop_top > 0:
            K1[1, 2] = K1[1, 2] - self.crop_top
            K2[1, 2] = K2[1, 2] - self.crop_top
            self.h = self.h - self.crop_top
        if self.crop_right > 0:
            self.w = self.w - self.crop_right
        if self.crop_bottom > 0:
            self.h = self.h - self.crop_bottom

        # Prepare rectification map
        R1_rect, R2_rect, P1_rect, P2_rect, _, _, _ = \
            cv2.stereoRectify(K1, D1, K2, D2, (self.w, self.h), R2, T2,
                              flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        self.map_x_l, self.map_y_l = cv2.initUndistortRectifyMap(
            K1, D1, R1_rect, P1_rect, (self.w, self.h), cv2.CV_32FC1
        )

        self.map_x_r, self.map_y_r = cv2.initUndistortRectifyMap(
            K2, D2, R2_rect, P2_rect, (self.w, self.h), cv2.CV_32FC1
        )

        K_rect = P1_rect[:, :3]
        # RT1_rect = np.linalg.inv(K_rect) @ P1_rect
        RT2_rect = np.matmul(np.linalg.inv(K_rect), P2_rect)

        # Rectified intrinsic and extrinsic params
        self.K = K_rect
        self.R = RT2_rect[:, :3]
        self.T = RT2_rect[:, 3]
        self.baseline = np.linalg.norm(self.T)

    def rectifyImages(self, img_left, img_right):
        if self.crop_top > 0:
            img_left = img_left[self.crop_top:, :, :]
            img_right = img_right[self.crop_top:, :, :]
        if self.crop_bottom > 0:
            img_left = img_left[:-self.crop_bottom, :, :]
            img_right = img_right[:-self.crop_bottom:, :, :]
        if self.crop_left > 0:
            img_left = img_left[:, self.crop_left:, :]
            img_right = img_right[:, self.crop_left:, :]
        if self.crop_right > 0:
            img_left = img_left[:, :-self.crop_right, :]
            img_right = img_right[:, :-self.crop_right, :]

        img_left_rect = cv2.remap(img_left, self.map_x_l, self.map_y_l,
                                  cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, self.map_x_r, self.map_y_r,
                                   cv2.INTER_LINEAR)

        return img_left_rect, img_right_rect
