import cv2
import time
import os
import json
from ultralytics import YOLO
import torch
import openai
import base64
import sys
import concurrent.futures


# 配置参数
class Config:
    # 模型配置
    MODEL_PATH = r'runs/train/exp/weights/best.pt'
    FALL_CLASS_ID = 0

    # 摄像头配置
    CAMERA_ID = 0  # 默认使用第一个摄像头，如果有多个摄像头可以修改这个值
    CAMERA_WIDTH = 640  # 摄像头分辨率宽度
    CAMERA_HEIGHT = 480  # 摄像头分辨率高度

    # 保存路径配置
    SAVE_FOLDER = r'C:\Users\Administrator\Desktop\yolo11-fall\yolo11-fall\ultralytics-8.3.70\results1'
    ALERT_FOLDER = os.path.join(SAVE_FOLDER, 'alerts')

    # API配置
    API_KEY = "sk-techtolwpjhplpxnuncitmuufucnrixywtbfabhyfrebgvpi"
    API_BASE = "https://api.siliconflow.cn/v1"

    # 检测参数
    FALL_DURATION_THRESHOLD = 3  # 跌倒持续时间阈值（秒）
    FRAME_SKIP = 4  # 每4帧进行一次推理
    CONSECUTIVE_FALL_FRAMES = 3  # 连续检测到跌倒的帧数阈值

    # 预警参数
    WARNING_COLOR = (0, 0, 255)  # 红色
    NORMAL_COLOR = (0, 255, 0)  # 绿色
    WARNING_TEXT = "Warning: Fall Detected!"

    # 模型推理参数
    INFERENCE_PARAMS = {
        'imgsz': 640,  # 增大图像大小以提高检测精度
        'conf': 0.3,  # 降低置信度阈值以检测更多可能的跌倒
        'iou': 0.5,  # 调整NMS阈值
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'half': False,
        'agnostic_nms': True,
        'max_det': 10,
        'save': False,
        'show': False,
        'project': 'runs/predict',
        'name': 'exp',
        'save_txt': False,
        'save_conf': True,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'vid_stride': 100,
        'line_width': 2,
        'visualize': False,
        'augment': False,
        'retina_masks': False,
        'boxes': True
    }


class VideoFallAnalyzer:
    def __init__(self):
        # 创建必要的文件夹
        os.makedirs(Config.SAVE_FOLDER, exist_ok=True)
        os.makedirs(Config.ALERT_FOLDER, exist_ok=True)

        # 检查CUDA是否可用
        self.device = Config.INFERENCE_PARAMS['device']
        print(f"使用设备: {self.device}")

        # 初始化YOLO模型
        self.model = YOLO(Config.MODEL_PATH)

        # 初始化API客户端
        self.client = openai.OpenAI(
            api_key=Config.API_KEY,
            base_url=Config.API_BASE
        )

        # 初始化摄像头
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头 {Config.CAMERA_ID}")

        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 初始化状态变量
        self.fall_start_time = None
        self.frame_count = 0
        self.is_warning = False
        self.consecutive_fall_frames = 0
        print("摄像头初始化完成，开始检测...")

    def analyze_with_gpt(self, alert_content):
        """使用 DeepSeek-R1 模型分析警报内容"""
        max_retries = 3
        retries = 0

        while retries < max_retries:
            try:
                print("正在调用 DeepSeek-R1 API...")
                prompt = f"""请分析以下跌倒警报信息，并提供约250字的详细分析和建议，涵盖以下方面：
1. 跌倒情况的严重程度评估
2. 可能的原因分析
3. 紧急处理建议
4. 预防措施建议
5. 后续跟进建议

请用专业且易懂的语言回答。

{alert_content}
"""

                print(f"API Key: {Config.API_KEY[:8]}...")
                print(f"API Base: {Config.API_BASE}")

                response = self.client.chat.completions.create(
                    model="Pro/Qwen/Qwen2.5-7B-Instruct",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                )

                # 获取分析结果
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
                content = getattr(response.choices[0].message, 'content', '')

                print("成功获取分析结果")
                return f"{reasoning_content}\n\n{content}"

            except openai.RateLimitError as e:
                print(f"API 速率限制错误: {str(e)}")
                print("等待 60 秒后重试...")
                time.sleep(60)
                retries += 1
            except openai.AuthenticationError as e:
                print(f"API 认证错误: {str(e)}")
                print("请检查 API 密钥是否正确")
                return "API 认证失败，请检查配置"
            except openai.APIConnectionError as e:
                print(f"API 连接错误: {str(e)}")
                print("请检查网络连接和 API 地址")
                return "API 连接失败，请检查网络"
            except Exception as e:
                print(f"分析错误: {str(e)}")
                print(f"错误类型: {type(e).__name__}")
                return f"分析出错: {str(e)}"

        return "分析失败，已达到最大重试次数"

    def save_alert(self, frame, analysis_result):
        """保存警报信息和分析结果"""
        timestamp = time.strftime("%Y年%m月%d日%H时%M分%S秒")

        # 保存图片
        image_name = f'fall_{timestamp}_{self.frame_count}.jpg'
        image_path = os.path.join(Config.ALERT_FOLDER, image_name)
        cv2.imwrite(image_path, frame)

        # 准备警报内容
        alert_content = f"""跌倒警报：
检测时间：{timestamp}
视频帧号：{self.frame_count}
事件描述：检测到老年人跌倒且持续时间超过{Config.FALL_DURATION_THRESHOLD}秒
老人糖尿病史 8 年，长期口服 "二甲双胍 0.5g tid"。
冠心病史 5 年，曾行冠脉支架植入术，规律服用 "阿司匹林肠溶片 100mg qd""阿托伐他汀 20mg qn"。
对青霉素过敏
建议操作：立即前往现场确认"""

        # 使用指定模型分析
        gpt_analysis = self.analyze_with_gpt(alert_content)

        # 保存完整的分析结果
        txt_name = f'fall_{timestamp}_{self.frame_count}.txt'
        txt_path = os.path.join(Config.ALERT_FOLDER, txt_name)

        complete_content = f"""{alert_content}

DeepseekR1分析结果：
{gpt_analysis}"""

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(complete_content)

        print(f"已保存警报图片: {image_path}")
        print(f"已保存分析文件: {txt_path}")

        # 释放资源
        self.cleanup()
        # 退出程序
        sys.exit(0)

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

    def draw_warning(self, frame):
        """在画面上绘制预警信息"""
        height, width = frame.shape[:2]

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 绘制预警文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(Config.WARNING_TEXT, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 40

        # 绘制文本阴影
        cv2.putText(frame, Config.WARNING_TEXT, (text_x + 1, text_y + 1),
                    font, font_scale, (0, 0, 0), thickness)
        # 绘制文本
        cv2.putText(frame, Config.WARNING_TEXT, (text_x, text_y),
                    font, font_scale, Config.WARNING_COLOR, thickness)

    def process_frame(self, frame):
        """处理单帧图像"""
        results = self.model.predict(
            source=frame,
            **Config.INFERENCE_PARAMS
        )

        fall_detected = False
        detection_result = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.cpu().numpy())
                confidence = box.conf.cpu().numpy()[0]

                if cls_id == Config.FALL_CLASS_ID:
                    fall_detected = True
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), Config.WARNING_COLOR, 2)
                    label = f'Fall {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, Config.WARNING_COLOR, 2)

                    detection_result.append({
                        'confidence': float(confidence),
                        'position': [int(x1), int(y1), int(x2), int(y2)]
                    })

        return fall_detected, detection_result

    def run(self):
        """运行检测循环"""
        print("开始视频检测，按 'q' 键退出...")

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("无法读取摄像头帧")
                        break

                    self.frame_count += 1

                    if self.frame_count % Config.FRAME_SKIP == 0:
                        future = executor.submit(self.process_frame, frame)
                        fall_detected, detection_result = future.result()

                        if fall_detected:
                            self.consecutive_fall_frames += 1
                            if self.consecutive_fall_frames >= Config.CONSECUTIVE_FALL_FRAMES:
                                if self.fall_start_time is None:
                                    self.fall_start_time = time.time()
                                    self.is_warning = True
                                else:
                                    elapsed_time = time.time() - self.fall_start_time
                                    if elapsed_time >= Config.FALL_DURATION_THRESHOLD:
                                        self.save_alert(frame, detection_result)
                        else:
                            self.consecutive_fall_frames = 0
                            self.fall_start_time = None
                            self.is_warning = False

                        # 显示预警信息
                        if self.is_warning:
                            self.draw_warning(frame)

                        # 显示处理后的帧
                        cv2.imshow('Fall Detection', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        finally:
            # 确保资源被正确释放
            self.cleanup()
            print("检测结束")


if __name__ == "__main__":
    try:
        analyzer = VideoFallAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        if 'analyzer' in locals():
            analyzer.cleanup()
