import cv2
import numpy as np
import json
import os


def json_to_mask(json_path, output_path, image_size=(340, 1700)):
    try:
        # 尝试读取 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {json_path} is not a valid JSON file.")
        return
    except PermissionError:
        print(f"Error: Permission denied when trying to access {json_path}. Please check file permissions.")
        return

    # 创建一个空白的 mask 图像，初始化为 0
    mask = np.zeros(image_size, dtype=np.uint8)

    if 'shapes' in data:
        for shape in data['shapes']:
            points = np.array(shape['points'], np.int32)
            if shape['shape_type'] == 'polygon':
                # 将多边形的点集转换为所需的格式
                points = points.reshape((-1, 1, 2))
                # 填充多边形区域
                cv2.fillPoly(mask, [points], 255)
            elif shape['shape_type'] == 'rectangle':
                # 处理矩形标注
                x1, y1 = int(shape['points'][0][0]), int(shape['points'][0][1])
                x2, y2 = int(shape['points'][1][0]), int(shape['points'][1][1])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            elif shape['shape_type'] == 'circle':
                # 处理圆形标注
                center = (int(shape['points'][0][0]), int(shape['points'][0][1]))
                radius = int(shape['radius'])
                cv2.circle(mask, center, radius, 255, -1)
            else:
                print(f"Warning: Unsupported shape type {shape['shape_type']}")
    else:
        print("Warning: No 'shapes' key found in the JSON data.")

    # 保存生成的 mask 图像
    try:
        cv2.imwrite(output_path, mask)
        print(f"Mask saved to {output_path}")
    except Exception as e:
        print(f"Error while saving the mask: {e}")


def process_directory(input_directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # 遍历目录中的文件
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                # 生成对应的输出文件路径
                output_file = os.path.splitext(file)[0] + '.png'
                output_path = os.path.join(output_directory, output_file)
                # 调用函数进行转换
                json_to_mask(json_path, output_path)


if __name__ == "__main__":
    input_directory = 'C:/Users/scjqw/Desktop/seg_mask/test'
    output_directory = 'C:/Users/scjqw/Desktop/seg_mask/test_mask'
    # 调用函数处理目录中的文件
    process_directory(input_directory, output_directory)

