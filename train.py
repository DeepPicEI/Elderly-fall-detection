
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'ultralytics/cfg/models/11/yolo11.yaml')  # 更改yolov11在自己电脑上的实际位置 这个文件夹在ultralytics\cfg\models\11\

    # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(
        data=r'my_fall.yaml',
        epochs=100,
        patience=50,
        batch=16,
        imgsz=640,
        save=True,
        save_period=-1,
        cache=False,
        device='',
        workers=0,
        project='runs/train',
        name='exp',
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        seed=0,
        deterministic=True,
        close_mosaic=0,
        resume=False,
        amp=True,
        fraction=1.0,
        freeze=None,

        # 超参数 ----------------------------------------------------------------------------------------------
        lr0=0.01,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
        lrf=0.01,  # (float) 最终学习率（lr0 * lrf）
        momentum=0.937,  # (float) SGD动量/Adam beta1
        weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
        warmup_epochs=3.0,  # (float) 预热周期（分数可用）
        warmup_momentum=0.8,  # (float) 预热初始动量
        warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
        box=7.5,  # (float) 盒损失增益
        cls=0.5,  # (float) 类别损失增益（与像素比例）
        dfl=1.5,  # (float) dfl损失增益
        pose=12.0,  # (float) 姿势损失增益
        kobj=1.0,  # (float) 关键点对象损失增益
        label_smoothing=0.0,  # (float) 标签平滑（分数）
        nbs=64,  # (int) 名义批量大小
        hsv_h=0.015,  # (float) 图像HSV-Hue增强（分数）
        hsv_s=0.7,  # (float) 图像HSV-Saturation增强（分数）
        hsv_v=0.4,  # (float) 图像HSV-Value增强（分数）
        degrees=0.0,  # (float) 图像旋转（+/- deg）
        translate=0.1,  # (float) 图像平移（+/- 分数）
        scale=0.5,  # (float) 图像缩放（+/- 增益）
        shear=0.0,  # (float) 图像剪切（+/- deg）
        perspective=0.0,  # (float) 图像透视（+/- 分数），范围为0-0.001
        flipud=0.0,  # (float) 图像上下翻转（概率）
        fliplr=0.5,  # (float) 图像左右翻转（概率）
        mosaic=1.0,  # (float) 图像马赛克（概率）
        mixup=0.0,  # (float) 图像混合（概率）
        copy_paste=0.0,  # (float) 分割复制-粘贴（概率）
    )

