from unittest import result
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy
import time
import torch

class ImageWatermarkRemover:
    def __init__(self):
        pass
    
    def parse_roi_string(self, roi_str, image_shape):
        """解析ROI字符串，格式为x1,y1,x2,y2（左上和右下坐标）"""
        try:
            x1, y1, x2, y2 = map(int, roi_str.split(','))
            # 验证坐标是否有效
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError("ROI坐标无效，确保左上坐标小于右下坐标且都为非负数")
            # 验证ROI是否在图像范围内
            if x2 > image_shape[1] or y2 > image_shape[0]:
                raise ValueError("ROI超出图像范围")
            # 转换为x,y,width,height格式用于后续处理
            x = x1
            y = y1
            width = x2 - x1
            height = y2 - y1
            return (x, y, width, height)
        except ValueError as e:
            raise ValueError(f"ROI格式错误，请使用x1,y1,x2,y2格式（左上和右下坐标）: {e}")
    
    def create_watermark_mask(self, image_shape, roi):
        """根据ROI创建水印掩码"""
        x, y, width, height = roi
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mask[y:y+height, x:x+width] = 255  # 水印区域设为白色
        return mask
    
    def preview_result(self, original, result, max_height=800):
        """预览处理结果"""
        h, w = result.shape[:2]
        scale_factor = max_height / h
        display_width = int(w * scale_factor)
        
        # 调整原始图像大小以匹配处理后的图像
        original_resized = cv2.resize(original, (display_width, int(max_height)))
        result_resized = cv2.resize(result, (display_width, int(max_height)))
        
        # 在图像上添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_resized, "Original", (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_resized, "Processed", (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 水平拼接图像
        combined = np.hstack((original_resized, result_resized))
        
        cv2.imshow("Original vs Processed", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_image_info(image_path):
    """获取图像信息"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        height, width = image.shape[:2]
        info = {
            "path": image_path,
            "resolution": f"{width}x{height}",
            "channels": image.shape[2] if len(image.shape) > 2 else 1
        }
        return info
    except Exception as e:
        raise Exception(f"获取图像信息出错: {e}")

def check_gpu():
    """检查GPU是否可用并返回设备信息"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        return device  # 返回设备字符串用于初始化模型
    else:
        return "cpu"

def initialize_lama(device="cpu"):
    # 启用PyTorch性能优化
    torch.backends.cudnn.benchmark = True  # 自动寻找最快的卷积算法
    
    # 初始化模型，移除不兼容的半精度转换
    model = ModelManager(name="lama", device=device)

    # 优化配置参数
    config = Config(
        ldm_steps=12,  # 减少采样步数，从25降至12，可大幅提升速度
        hd_strategy=HDStrategy.CROP,  # 使用裁剪策略而非处理整个图像
        hd_strategy_crop_margin=64,  # 增加裁剪边缘以获得更好的上下文
        hd_strategy_crop_trigger_size=1024,  # 降低触发裁剪的阈值
        hd_strategy_resize_limit=1024,  # 降低调整大小的限制
    )

    return model, config

def optimize_image_for_inference(image, mask, margin=64):
    """优化图像以加速推理，只保留水印区域及其周围的内容"""
    # 找到包含水印的区域边界
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return image, mask, (0, 0, image.shape[1], image.shape[0])
    
    # 计算水印区域的边界框
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 添加安全边距并确保不超出图像边界
    h, w = image.shape[:2]
    y_min = max(0, y_min - margin)
    y_max = min(h, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(w, x_max + margin)
    
    # 裁剪图像和掩码到水印区域
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    
    return cropped_image, cropped_mask, (x_min, y_min, x_max, y_max)

def lama_inpaint(frame, mask, model, config):
    # 优化图像，只处理水印区域及其周围
    cropped_frame, cropped_mask, (x_min, y_min, x_max, y_max) = optimize_image_for_inference(frame, mask)
    
    mask_binary = np.where(cropped_mask > 0, 255, 0).astype(np.uint8)
    
    # 只对裁剪后的区域进行推理
    result_cropped = model(cropped_frame, mask_binary, config)
    
    # 确保结果格式正确
    if result_cropped.dtype == np.float64:
        if np.max(result_cropped) <= 1.0:
            result_cropped = (result_cropped * 255).astype(np.uint8)
        else:
            result_cropped = result_cropped.astype(np.uint8)
    
    # 创建完整的结果图像，只替换水印区域
    result_full = frame.copy()
    result_cropped = cv2.cvtColor(result_cropped, cv2.COLOR_RGB2BGR)
    result_full[y_min:y_max, x_min:x_max] = result_cropped
    
    return result_full

def ensure_directory_exists(directory):
    """确保目录存在且可写"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return True
        except OSError as error:
            print(f"创建目录错误 {directory}: {error}")
            return False
    
    temp_file = os.path.join(directory, f"temp_{time.time()}.tmp")
    try:
        with open(temp_file, 'w') as f:
            f.write("test")
        os.remove(temp_file)
        return True
    except Exception as e:
        print(f"目录无写权限 {directory}: {e}")
        return False

def is_valid_image_file(file_path):
    """检查文件是否为有效的图像文件"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    if not os.path.isfile(file_path):
        return False
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_extensions:
        return False
    
    try:
        img = cv2.imread(file_path)
        return img is not None
    except Exception as e:
        print(f"无效图像文件: {file_path}, 错误: {e}")
        return False

def process_image(input_path, roi_str, output_path=None, preview=True, inpainting_model=None):
    """处理单张图片，移除指定区域的水印"""
    # 检查输入文件
    if not is_valid_image_file(input_path):
        raise ValueError(f"无效的图像文件: {input_path}")
    
    # 读取图像
    image = cv2.imread(input_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 初始化图像水印移除器
    remover = ImageWatermarkRemover()
    
    # 解析ROI
    roi = remover.parse_roi_string(roi_str, image.shape)
    
    # 创建水印掩码
    watermark_mask = remover.create_watermark_mask(image.shape, roi)
    
    # 初始化模型
    if inpainting_model is None:
        device = check_gpu()
        inpainting_model = initialize_lama(device)
    
    model, config = inpainting_model
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行修复
    result_bgr = lama_inpaint(image_rgb, watermark_mask, model, config)
        # 计算处理时间
    processing_time = time.time() - start_time

    
    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = 'output'
        output_path = os.path.join(output_dir, f"{base_name}_no_watermark.jpg")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not ensure_directory_exists(output_dir):
        raise Exception(f"无法创建输出目录: {output_dir}")
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    # 保存结果
    if not cv2.imwrite(output_path, result_rgb):
        raise Exception(f"无法保存结果到: {output_path}")
    
    # 预览结果
    if preview:
        remover.preview_result(image, result_rgb)
    
    # 收集处理信息
    processing_info = {
        "input_path": input_path,
        "output_path": output_path,
        "roi": roi_str,
        "processing_time": processing_time
    }
    
    return processing_info

def get_config():
    # 直接在代码中配置参数
    config = {
        "input_path": "D:\CODE\WatermarkRemover-master\image",  # 输入图片路径
        "roi": "1281,2750,1580,2822",  # 水印区域坐标 (x1,y1,x2,y2)，即左上和右下坐标
        "output_path": None, #"image/no_watermark.jpg",  # 输出图片路径
        "preview": False  # 是否显示预览
    }
    return config

if __name__ == "__main__":
    print("=== 图片水印移除工具 ===")
    print("参数配置在代码中，请修改get_config()函数中的相关变量")
    
    # 获取配置
    config = get_config()
    input_path = config["input_path"]
    roi_str = config["roi"]
    output_path = config["output_path"]
    preview = config["preview"]
    
    print(f"当前配置:")
    print(f"  输入图片: {input_path}")
    print(f"  水印区域: {roi_str} (x1,y1,x2,y2)，即左上和右下坐标")
    print(f"  输出图片: {output_path}")
    print(f"  显示预览: {preview}")
    print("\n开始处理...")

    # 检查GPU可用性
    device = check_gpu()
    print(f"使用设备: {device}")
    
    # 初始化修复模型
    lama_model = initialize_lama(device)
    
    try:
        # 检查输入路径是文件还是文件夹
        if os.path.isfile(input_path):
            # 处理单个文件
            processing_info = process_image(
                input_path=input_path,
                roi_str=roi_str,
                output_path=output_path,
                preview=preview,
                inpainting_model=lama_model
            )
            
            print("\n处理完成!")
            print(f"  输入路径: {processing_info['input_path']}")
            print(f"  输出路径: {processing_info['output_path']}")
            print(f"  水印区域: {processing_info['roi']} (x1,y1,x2,y2)")
            print(f"  处理时间: {processing_info['processing_time']}")
        else:
            # 处理文件夹中的所有图片
            print(f"\n开始批量处理文件夹: {input_path}")
            total_processed = 0
            total_time = 0
            
            # 获取文件夹中的所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = [f for f in os.listdir(input_path) 
                          if any(f.lower().endswith(ext) for ext in image_extensions)]
            
            print(f"找到 {len(image_files)} 个图片文件")
            
            # 为每个图片文件调用process_image
            for i, image_file in enumerate(image_files, 1):
                image_path = os.path.join(input_path, image_file)
                print(f"\n[{i}/{len(image_files)}] 处理文件: {image_file}")
            
                try:
                    processing_info = process_image(
                        input_path=image_path,
                        roi_str=roi_str,
                        output_path=output_path,
                        preview=preview,
                        inpainting_model=lama_model
                    )
                    
                    print(f"  输出路径: {processing_info['output_path']}")
                    print(f"  处理时间: {processing_info['processing_time']}")
                    total_processed += 1
                    total_time += processing_info['processing_time']
                except Exception as file_error:
                    print(f"  处理失败: {file_error}")
            
            print("\n批量处理完成!")
            print(f"  成功处理: {total_processed}/{len(image_files)} 个文件")
            print(f"  总处理时间: {total_time:.2f} 秒")
            if total_processed > 0:
                print(f"  平均处理时间: {total_time/total_processed:.2f} 秒/文件")
                
    except Exception as e:
        print(f"处理过程中出错: {e}")
        sys.exit(1)