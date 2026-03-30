import os
import glob
import argparse
from pathlib import Path


def collect_images_from_directories(directories, extensions):
    """
    从多个目录中收集所有指定扩展名的图像文件
    
    Args:
        directories: 目录路径列表
        extensions: 图像扩展名列表
    
    Returns:
        排序后的图像绝对路径列表
    """
    image_paths = []
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"警告: 目录不存在 - {directory}")
            continue
        
        if not dir_path.is_dir():
            print(f"警告: 不是目录 - {directory}")
            continue
        
        # 遍历每种扩展名
        for ext in extensions:
            # 支持大小写不敏感的扩展名匹配
            pattern = f"**/*.{ext}"
            found_images = list(dir_path.glob(pattern))
            
            # 也尝试大写扩展名
            pattern_upper = f"**/*.{ext.upper()}"
            found_images.extend(list(dir_path.glob(pattern_upper)))
            
            # 转换为绝对路径
            for img_path in found_images:
                abs_path = str(img_path.resolve())
                if abs_path not in image_paths:  # 避免重复
                    image_paths.append(abs_path)
    
    # 排序
    image_paths.sort()
    
    return image_paths


def write_image_list_to_txt(image_paths, output_file):
    """
    将图像路径列表写入txt文件
    
    Args:
        image_paths: 图像路径列表
        output_file: 输出txt文件路径
    """
    output_path = Path(output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for img_path in image_paths:
            f.write(f"{img_path}\n")
    
    print(f"成功写入 {len(image_paths)} 个图像路径到 {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='遍历多个目录，收集图像文件路径并写入txt文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python gen_image_list_txt.py -d /path/to/dir1 /path/to/dir2 -o output.txt
  python gen_image_list_txt.py -d /path/to/dir1 -o output.txt -e jpg png jpeg
        """
    )
    
    parser.add_argument(
        '-d', '--directories',
        nargs='+',
        required=True,
        help='要遍历的目录路径列表（可以传入多个目录，用空格分隔）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出txt文件的路径'
    )
    
    parser.add_argument(
        '-e', '--extensions',
        nargs='+',
        default=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        help='图像文件扩展名列表（默认: png jpg jpeg bmp tiff webp）'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='是否递归遍历子目录（默认: True）'
    )
    
    args = parser.parse_args()
    
    print(f"开始处理...")
    print(f"目录列表: {args.directories}")
    print(f"图像扩展名: {args.extensions}")
    print(f"输出文件: {args.output}")
    print("-" * 50)
    
    # 收集图像
    image_paths = collect_images_from_directories(args.directories, args.extensions)
    
    if not image_paths:
        print("警告: 没有找到任何图像文件！")
        return
    
    print(f"找到 {len(image_paths)} 个图像文件")
    
    # 写入文件
    write_image_list_to_txt(image_paths, args.output)
    
    print("完成！")


if __name__ == "__main__":
    main()
