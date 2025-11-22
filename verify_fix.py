import cv2
import numpy as np
from src.backend.recognizer import HandwrittenRecognizer
from src.backend.preprocessor import ImagePreprocessor

# 创建一个更完整的测试脚本
print("=== 验证修复后的手写数字识别功能 ===")

# 创建识别器和预处理器
recognizer = HandwrittenRecognizer()
preprocessor = ImagePreprocessor()

# 测试函数：创建数字图像并测试

def test_digit(digit):
    print(f"\n测试数字: {digit}")
    # 创建一个更大的图像，然后缩小，以获得更好的数字形状
    large_img = np.ones((200, 200), dtype=np.uint8) * 255  # 白色背景
    
    # 写入数字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 10
    text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
    text_x = (200 - text_size[0]) // 2
    text_y = (200 + text_size[1]) // 2
    cv2.putText(large_img, str(digit), (text_x, text_y), font, font_scale, 0, thickness)
    
    # 缩小到28x28
    img = cv2.resize(large_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 预处理图像
    processed = preprocessor.preprocess(img)
    
    if processed is not None:
        # 进行识别
        result, confidence = recognizer.recognize(processed)
        
        # 评估结果 - 确保类型匹配
        is_correct = int(result) == digit
        status = "✓ 正确识别" if is_correct else "✗ 识别错误"
        
        print(f"{status}")
        print(f"预测结果: {result}, 实际数字: {digit}")
        print(f"置信度: {confidence:.4f}")
        
        return is_correct
    else:
        print("预处理失败")
        return False

# 测试所有数字
correct_count = 0
total_count = 10

for digit in range(total_count):
    if test_digit(digit):
        correct_count += 1

# 打印测试统计
print("\n=== 测试统计 ===")
print(f"总测试数: {total_count}")
print(f"正确识别数: {correct_count}")
print(f"准确率: {correct_count / total_count * 100:.1f}%")

if correct_count >= total_count * 0.7:  # 70%以上认为修复成功
    print("\n✓ 修复成功！模型现在能够正确识别大部分数字。")
else:
    print("\n✗ 修复不完全，请进一步检查问题。")
