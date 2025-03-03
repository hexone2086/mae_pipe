#!/home/hexone/ros2_ws/src/mae_pipe/mae_pipe/venv/bin/python3

import struct
import numpy as np
import time
import cv2
import lz4.frame
import json

import rano
from typing import List, Tuple, Union
import pickle
import os
import threading

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from loguru import logger

import sys

# 定义常量
RGB_IMAGE_TOPIC = '/camera/rgb/image_raw'
DEPTH_IMAGE_TOPIC = '/camera/depth/image_raw'

frame = None
DynamicMask = False

# 添加模块路径
sys.path.append('/home/lyk/1/mae_pipe')
import serial_module

image_stds = np.random.rand(49).astype(np.float16) # 第一个 f32 数组
image_means = np.random.rand(49).astype(np.float16) # 第二个 f32 数组
mask = np.random.choice([True, False], size=50) # bool 数组

std_max = 0
std_min = 100
mean_max = 0
mean_min = 100

frame_idx = 0
timestamp_base = time.time()

control_dict = None

def pack_data(jpeg_data, timestamp, image_means, image_stds, mask):
    # 将数据拼接为二进制格式
    data_to_compress = (
        jpeg_data.tobytes() +
        timestamp.tobytes() +
        image_means.tobytes() +
        image_stds.tobytes() +
        mask.tobytes()
    )

    # 使用 lz4 压缩数据
    compressed_data = lz4.frame.compress(data_to_compress)

    # 创建元数据
    metadata = {
        'jpeg_data_size': jpeg_data.nbytes,
        'timestamp_size': timestamp.nbytes,
        'image_means_size': image_means.nbytes,
        'image_stds_size': image_stds.nbytes,
        'mask_size': mask.nbytes,
    }

    # 将元数据序列化为 JSON 字符串
    metadata_json = json.dumps(metadata).encode('utf-8')

    # 将元数据和压缩后的数据拼接在一起
    packed_data = len(metadata_json).to_bytes(4, 'little') + metadata_json + compressed_data

    return packed_data

def linear_map_to_uint8(mean_array, std_array):
    # 均值映射
    mean_min, mean_max = 0.0, 1.0
    quantized_mean = np.round((mean_array - mean_min) / (mean_max - mean_min) * 255.0).astype(np.uint8)
    
    # 标准差非线性映射
    quantized_std = np.round((1.0 - np.exp(-std_array * -(30))) * 255.0).astype(np.uint8)

    return quantized_mean, quantized_std

# 初始化串口
while True:
    try:
        ser = serial_module.init("/dev/ttyUSB0")
        print(f'serial ok {ser}')
        break
    except Exception as e:
        print(f'serial not ok: {e}')
        time.sleep(1)

# 定义回调函数
def rgb_image_callback(msg):
    global frame
    bridge = CvBridge()
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # 转换为 OpenCV 图像
        # logger.info("Received RGB image")
    except Exception as e:
        logger.error(f"Error converting RGB image: {e}")

def depth_image_callback(msg):
    bridge = CvBridge()
    try:
        depth_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1') # 深度图像通常为 32 位浮点数
        logger.info("Received depth image")
    except Exception as e:
        logger.error(f"Error converting depth image: {e}")

# ROS1初始化
rospy.init_node('mae_pipe', anonymous=True)
rospy.loginfo(f"Subscribed to RGB topic: {RGB_IMAGE_TOPIC}")
rospy.loginfo(f"Subscribed to depth topic: {DEPTH_IMAGE_TOPIC}")

# 订阅RGB和深度图像
rgb_sub = rospy.Subscriber(RGB_IMAGE_TOPIC, Image, rgb_image_callback, queue_size=10)
depth_sub = rospy.Subscriber(DEPTH_IMAGE_TOPIC, Image, depth_image_callback, queue_size=10)

def node_thread():
    try:
        rospy.spin()  # ROS1中的spin
    except KeyboardInterrupt:
        pass

threading.Thread(target=node_thread).start()

frame_remain = None
frame_entire = None

while True:
    if frame is None:
        time.sleep(1)
        continue

    frame_entire = cv2.resize(frame, (224, 224))

    if control_dict:
        x1 = control_dict['roi_x1']
        y1 = control_dict['roi_y1']
        x2 = control_dict['roi_x2']
        y2 = control_dict['roi_y2']
        mode = control_dict['mode']
        frame_remain, mask = rano.generate((x1, y1, x2 - x1, y2 - y1), frame_entire, mode=mode)
    else:
        frame_remain, mask = rano.generate((0, 0, 224, 224), frame_entire, mode='auto-fixed')

    cv2.imshow('TX Realtime Preview (masked)', rano.restore_image_from_patches(frame_remain, mask, (224, 224)))
    cv2.waitKey(1)

    # 将帧转换为 JPEG 格式的二进制数据
    _, jpeg_data = cv2.imencode('.jpg', frame_remain, [cv2.IMWRITE_JPEG_QUALITY, 20])

    timestamp = np.array([time.time() - timestamp_base], dtype=np.float64)
    frame_idx += 1

    _, image_stds = rano.calculate_patch_mean_variance(cv2.cvtColor(frame_entire, cv2.COLOR_BGR2RGB))
    image_means, _ = rano.calculate_patch_mean_variance_old(cv2.cvtColor(frame_entire, cv2.COLOR_BGR2RGB))
    image_stds_tiny = rano.reduce_to_large_patch(image_stds)

    image_means, _ = linear_map_to_uint8(image_means, image_stds)

    # 打包并发送数据
    packed_data = pack_data(jpeg_data, timestamp, image_means, image_stds_tiny, mask)

    logger.debug(f"size jpeg_data: {jpeg_data.nbytes}, size timestamp: {timestamp.nbytes}, size image_means: {image_means.nbytes}, size image_stds: {image_stds.nbytes}, size mask: {mask.nbytes}")

    ret = serial_module.send(packed_data)
    if ret is False:
        logger.warning(f"send frame {frame_idx} with NACK")
    else:
        msg_ack = tuple(ret) # ret 为 ack 二进制数据，需转换成元组
        logger.success(f"send frame {frame_idx} with ACK {len(msg_ack)} {msg_ack[1]}")