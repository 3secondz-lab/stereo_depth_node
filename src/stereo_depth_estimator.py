#!/usr/bin/env python
import os

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import cv2
import numpy as np

from common import StereoImagePreprocessor

# Global variables
preprocessor = None
rate = None
stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=31,
    P1=8*31*31, P2=32*31*31
)


def callback(msg_left, msg_right):
    global preprocessor, rate, stereo_matcher

    t_left = msg_left.header.stamp.to_nsec()
    t_right = msg_right.header.stamp.to_nsec()

    t_diff = abs(t_left - t_right)

    rospy.loginfo('Time difference : {:10.6f} ms'.format(t_diff / 1000000.0))

    img_l = cv2.imdecode(np.fromstring(msg_left.data, np.uint8),
                         cv2.IMREAD_UNCHANGED)
    img_r = cv2.imdecode(np.fromstring(msg_right.data, np.uint8),
                         cv2.IMREAD_UNCHANGED)

    img_l_rect, img_r_rect = preprocessor.rectifyImages(img_l, img_r)

    # img_l_gray = cv2.cvtColor(img_l_rect, cv2.COLOR_BGR2GRAY)
    # img_r_gray = cv2.cvtColor(img_r_rect, cv2.COLOR_BGR2GRAY)

    disp = stereo_matcher.compute(img_l_rect, img_r_rect)
    # disp = stereo_matcher.compute(img_l_gray, img_r_gray)
    mask = disp > 0

    disp = disp.astype(np.float32) / 16.0
    disp = (disp / 128.0 * 255.0).astype(np.uint8)
    disp = disp * mask


    # import pdb
    # pdb.set_trace()

    cv2.namedWindow("img")
    cv2.imshow("img", np.concatenate([img_l_rect, img_r_rect], axis=1))
    cv2.namedWindow("disp")
    cv2.imshow("disp", disp)
    cv2.waitKey(1)

    rate.sleep()


def main():
    global preprocessor, rate

    name_node = 'stereo_depth_estimator'

    rospy.init_node(name_node, anonymous=False)

    topic_left = rospy.get_param('~topic_left', '/cam0/pub/image/compressed')
    topic_right = rospy.get_param('~topic_right', '/cam1/pub/image/compressed')

    path_calib_left = rospy.get_param('~calib_left', '')
    path_calib_right = rospy.get_param('~calib_right', '')

    width = rospy.get_param('~width', 1280)
    height = rospy.get_param('~height', 1024)

    crop_top = rospy.get_param('~crop_top', 350)
    crop_bottom = rospy.get_param('~crop_bottom', 350)
    crop_left = rospy.get_param('~crop_left', 0)
    crop_right = rospy.get_param('~crop_right', 0)

    queue_size = rospy.get_param('~queue_size', 3)
    freq = rospy.get_param('freq', 20)

    t_diff_max = 2.0 / freq

    rospy.loginfo('Left : {}'.format(topic_left))
    rospy.loginfo('Right : {}'.format(topic_right))

    rospy.loginfo('Calib Left : {}'.format(path_calib_left))
    rospy.loginfo('Calib Right : {}'.format(path_calib_right))

    rospy.loginfo('Image Size : [{}, {}]'.format(width, height))

    rospy.loginfo('Crop Top : {}'.format(crop_top))
    rospy.loginfo('Crop Bottom : {}'.format(crop_bottom))
    rospy.loginfo('Crop Left : {}'.format(crop_left))
    rospy.loginfo('Crop Right : {}'.format(crop_right))

    rospy.loginfo('Queue Size : {}'.format(queue_size))
    rospy.loginfo('Freq : {}'.format(freq))
    rospy.loginfo('Max Time Difference : {}'.format(t_diff_max))

    sub_left = message_filters.Subscriber(topic_left, CompressedImage,
                                          queue_size=queue_size)
    sub_right = message_filters.Subscriber(topic_right, CompressedImage,
                                           queue_size=queue_size)

    ts = message_filters.ApproximateTimeSynchronizer(
        [sub_left, sub_right], queue_size=queue_size, slop=t_diff_max)

    ts.registerCallback(callback)

    preprocessor = StereoImagePreprocessor(
        path_calib_left, path_calib_right, width, height,
        crop_top, crop_bottom, crop_left, crop_right
    )

    rate = rospy.Rate(freq)

    rospy.spin()


if __name__ == '__main__':
    main()
