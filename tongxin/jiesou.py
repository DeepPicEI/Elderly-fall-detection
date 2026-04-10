# -*- coding: gbk -*-
# receiver_pc.py - 电脑端：接收视频流并显示

import cv2
import socket
import struct
import pickle

# 监听IP和端口
host = '0.0.0.0'  # 接收任意地址发来的连接
port = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print("等待树莓派连接中...")

conn, addr = server_socket.accept()
print('连接来自:', addr)
data = b''
payload_size = struct.calcsize(">L")

try:
    while True:
        # 获取数据长度
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_size)[0]

        # 获取完整数据
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # 解码图像并显示
        frame = pickle.loads(frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        cv2.imshow("接收到的视频", frame)

        # 按键退出
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # 关闭连接
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()