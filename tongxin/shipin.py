# sender_pi.py - 树莓派端：采集摄像头画面并发送
# -*- coding: utf-8 -*-

import cv2
import socket
import struct
import pickle

# 设置电脑的IP地址和端口
server_ip = '192.168.43.125'  # 换成你电脑的局域网IP
server_port = 9999

# 建立socket连接
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))
connection = client_socket.makefile('wb')

# 打开摄像头（0表示默认摄像头）
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 压缩图像，减少传输量
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        data = pickle.dumps(buffer)
        size = len(data)

        # 发送长度和数据
        client_socket.sendall(struct.pack(">L", size) + data)

finally:
    cap.release()
    connection.close()
    client_socket.close()