#!/usr/bin/env python
import os
import sys
import math

sys.path.insert(0, '/home/ys/Research/stereo_depth_node/src/thirdparty/aanet')

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import cv2
import numpy as np
from easydict import EasyDict as edict

import torch
from common import StereoImagePreprocessor
import thirdparty.aanet.nets as nets
from thirdparty.aanet.dataloader import transforms
from thirdparty.aanet.utils import utils

# Global variables
params = edict()


def callback(msg_left, msg_right):
    global params

    t_left = msg_left.header.stamp.to_nsec()
    t_right = msg_right.header.stamp.to_nsec()

    t_diff = abs(t_left - t_right)

    rospy.loginfo('Time difference : {:10.6f} ms'.format(t_diff / 1000000.0))
    rospy.loginfo('ID : {} / {}'.format(msg_left.header.seq, msg_right.header.seq))

    img_l = cv2.imdecode(np.fromstring(msg_left.data, np.uint8),
                         cv2.IMREAD_UNCHANGED)
    img_r = cv2.imdecode(np.fromstring(msg_right.data, np.uint8),
                         cv2.IMREAD_UNCHANGED)

    img_l_rect_o, img_r_rect_o = params.preprocessor.rectifyImages(img_l, img_r)

    if params.algorithm == 'SGBM':
        scale_factor = params.scale_factor

        h_new = int(img_l_rect_o.shape[0] * params.scale_factor)
        w_new = int(img_l_rect_o.shape[1] * params.scale_factor)

        img_l_rect = cv2.resize(img_l_rect_o, (w_new, h_new))
        img_r_rect = cv2.resize(img_r_rect_o, (w_new, h_new))

        img_l_rect = cv2.cvtColor(img_l_rect, cv2.COLOR_BGR2GRAY)
        img_r_rect = cv2.cvtColor(img_r_rect, cv2.COLOR_BGR2GRAY)

        disp = params.stereo_matcher.compute(img_l_rect, img_r_rect)
        disp = disp.astype(np.float32) / 16.0
    elif params.algorithm == 'AANet':
        h_new = img_l_rect_o.shape[0] * params.scale_factor
        w_new = img_l_rect_o.shape[1] * params.scale_factor

        h_new = int(math.ceil(h_new / 48) * 48)
        w_new = int(math.ceil(w_new / 48) * 48)

        scale_factor = w_new / img_l_rect_o.shape[1]

        img_l_rect = cv2.resize(img_l_rect_o, (w_new, h_new))
        img_r_rect = cv2.resize(img_r_rect_o, (w_new, h_new))

        img_l_rect = img_l_rect.astype(np.float32) / 255.0
        img_r_rect = img_r_rect.astype(np.float32) / 255.0

        sample = {'left': img_l_rect, 'right': img_r_rect}
        sample = params.aanet_transform(sample)

        img_l_rect = sample['left'].cuda().unsqueeze(0)
        img_r_rect = sample['right'].cuda().unsqueeze(0)

        with torch.no_grad():
            disp = params.aanet(img_l_rect, img_r_rect)[-1]
            disp = disp[0, :, :].detach().cpu().numpy()
    else:
        rospy.logerr('Unrecognized algorithm : {}'.format(params.algorithm))
        return

    depth = params.preprocessor.K[0, 0] * params.preprocessor.baseline * scale_factor / (disp + 1e-8)

    mask_valid = ((disp > 0) * (depth > 0) * (depth <= params.max_depth)).astype(np.float32)

    depth = depth * mask_valid

    depth_vis = (255.0 * depth / params.max_depth).astype(np.uint8)
    depth_vis = mask_valid[:, :, np.newaxis].astype(np.uint8) * cv2.applyColorMap(depth_vis, cv2.COLORMAP_PARULA)

    h, w = img_l_rect_o.shape[0:2]
    depth = cv2.resize(depth, (w, h))
    depth_vis = cv2.resize(depth_vis, (w, h))

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", np.concatenate([img_l_rect_o, img_r_rect_o], axis=1))
    cv2.namedWindow("depth_vis", cv2.WINDOW_NORMAL)
    cv2.imshow("depth_vis", depth_vis)
    cv2.waitKey(1)

    params.rate.sleep()


def main():
    global params

    name_node = 'stereo_depth_estimator'

    rospy.init_node(name_node, anonymous=False)

    topic_left = rospy.get_param('~topic_left', '/cam0/pub/image/compressed')
    topic_right = rospy.get_param('~topic_right', '/cam1/pub/image/compressed')

    path_calib_left = rospy.get_param('~calib_left', '/home/ys/catkin_ws/src/stereo_depth_node/calib/cam0.xml')
    path_calib_right = rospy.get_param('~calib_right', '/home/ys/catkin_ws/src/stereo_depth_node/calib/cam1.xml')

    params.width = rospy.get_param('~width', 1280)
    params.height = rospy.get_param('~height', 1024)

    params.scale_factor = rospy.get_param('~scale_factor', 1.0)

    params.crop_top = rospy.get_param('~crop_top', 350)
    params.crop_bottom = rospy.get_param('~crop_bottom', 350)
    params.crop_left = rospy.get_param('~crop_left', 0)
    params.crop_right = rospy.get_param('~crop_right', 0)

    queue_size = rospy.get_param('~queue_size', 10)
    freq = rospy.get_param('freq', 20)

    params.min_disp = rospy.get_param('~min_disp', 0)
    params.num_disp = rospy.get_param('~num_disp', 64)
    params.block_size = rospy.get_param('~block_size', 15)

    params.max_depth = rospy.get_param('~max_depth', 100.0)

    # params.pretrained = rospy.get_param('~pretrained', '/home/ys/Research/stereo_depth_node/src/thirdparty/aanet/pretrained/aanet+_kitti15-2075aea1.pth')
    params.pretrained = rospy.get_param('~pretrained', '/home/ys/Research/stereo_depth_node/src/thirdparty/aanet/pretrained/aanet_kitti15-fb2a0d23.pth')

    params.algorithm = rospy.get_param('~algorithm', 'AANet')

    t_diff_max = 0.5 / freq

    rospy.loginfo('Left : {}'.format(topic_left))
    rospy.loginfo('Right : {}'.format(topic_right))

    rospy.loginfo('Calib Left : {}'.format(path_calib_left))
    rospy.loginfo('Calib Right : {}'.format(path_calib_right))

    rospy.loginfo('Image Size : [{}, {}]'.format(params.width, params.height))

    rospy.loginfo('Crop Top : {}'.format(params.crop_top))
    rospy.loginfo('Crop Bottom : {}'.format(params.crop_bottom))
    rospy.loginfo('Crop Left : {}'.format(params.crop_left))
    rospy.loginfo('Crop Right : {}'.format(params.crop_right))

    rospy.loginfo('Queue Size : {}'.format(queue_size))
    rospy.loginfo('Freq : {}'.format(freq))
    rospy.loginfo('Max Time Difference : {}'.format(t_diff_max))

    rospy.loginfo('Algorithm : {}'.format(params.algorithm))

    params.preprocessor = StereoImagePreprocessor(
        path_calib_left, path_calib_right, params.width, params.height,
        params.crop_top, params.crop_bottom, params.crop_left, params.crop_right
    )

    if params.algorithm == 'SGBM':
        params.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=params.min_disp, numDisparities=params.num_disp, blockSize=params.block_size,
            P1=8*params.block_size*params.block_size, P2=32*params.block_size*params.block_size
        )
    elif params.algorithm == 'AANet':
        # AANet
        params.imagenet_mean = [0.485, 0.456, 0.406]
        params.imagenet_std = [0.229, 0.224, 0.225]
        params.aanet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.imagenet_mean, std=params.imagenet_std
            )
        ])

        # params.aanet = nets.AANet(
        #     192,
        #     num_downsample=2,
        #     feature_type='ganet',
        #     no_feature_mdconv=False,
        #     feature_pyramid=True,
        #     feature_pyramid_network=False,
        #     feature_similarity='correlation',
        #     aggregation_type='adaptive',
        #     num_scales=3,
        #     num_fusions=6,
        #     num_stage_blocks=1,
        #     num_deform_blocks=3,
        #     no_intermediate_supervision=True,
        #     refinement_type='hourglass',
        #     mdconv_dilation=2,
        #     deformable_groups=2).cuda().eval()

        params.aanet = nets.AANet(
            192,
            num_downsample=2,
            feature_type='aanet',
            no_feature_mdconv=False,
            feature_pyramid=False,
            feature_pyramid_network=True,
            feature_similarity='correlation',
            aggregation_type='adaptive',
            num_scales=3,
            num_fusions=6,
            num_stage_blocks=1,
            num_deform_blocks=3,
            no_intermediate_supervision=True,
            refinement_type='stereodrnet',
            mdconv_dilation=2,
            deformable_groups=2).cuda()

        utils.load_pretrained_net(params.aanet, params.pretrained, no_strict=True)
        params.aanet.eval()
    else:
        rospy.logerr('Unrecognized algorithm : {}'.format(params.algorithm))
        return

    params.rate = rospy.Rate(freq)

    sub_left = message_filters.Subscriber(topic_left, CompressedImage,
                                          queue_size=queue_size)
    sub_right = message_filters.Subscriber(topic_right, CompressedImage,
                                           queue_size=queue_size)
    
    ts = message_filters.ApproximateTimeSynchronizer(
        [sub_left, sub_right], queue_size=3, slop=t_diff_max)

    ts.registerCallback(callback)

    rospy.spin()


if __name__ == '__main__':
    main()
