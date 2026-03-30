import base64
import logging
import mimetypes
import os
from io import BytesIO

import dashscope
import numpy as np
import requests
from dashscope import MultiModalConversation
from PIL import Image

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 设置API基础URL
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

class QwenImageEdit:
    """Qwen图像编辑类"""
    
    def __init__(self, api_key=None, base_url=None):
        """
        初始化Qwen图像编辑器
        
        Args:
            api_key: API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY获取
            base_url: API基础URL，默认为中国（北京）地域
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请提供API密钥或设置环境变量DASHSCOPE_API_KEY")
        
        # 设置API基础URL
        if base_url:
            dashscope.base_http_api_url = base_url
        
        logger.info("Qwen图像编辑器初始化完成")

    def encode_file(self, file_path):
        """
        将图像文件编码为Base64格式
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            str: Base64编码的图像数据
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"图像文件不存在: {file_path}")

        # 检查是否为文件（而不是目录）
        if not os.path.isfile(file_path):
            raise ValueError(f"路径不是文件: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("不支持或无法识别的图像格式")

        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
        except IOError as e:
            raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

    def encode_image_object(self, image):
        """
        将PIL Image对象或numpy数组编码为Base64格式
        
        Args:
            image: PIL Image对象或numpy数组
            
        Returns:
            str: Base64编码的图像数据
        """
        try:
            # 处理不同类型的图像输入
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image)
            else:
                pil_img = image
            
            # 转换为RGB格式（如果需要）
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # 编码为Base64
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{encoded_string}"
            
        except Exception as e:
            raise ValueError(f"图像编码失败: {str(e)}")

    def edit_image(self, image, prompt, negative_prompt="", model="qwen-image-edit"):
        """
        编辑图像
        
        Args:
            image: 图像文件路径、PIL Image对象或numpy数组
            prompt: 编辑指令
            negative_prompt: 负面提示词（可选）
            model: 模型名称，默认为qwen-image-edit
            
        Returns:
            dict: 编辑结果
        """
        try:
            logger.info(f"开始图像编辑 | 模型: {model}")
            
            # 处理图像输入
            if isinstance(image, str):
                # 文件路径
                image_data = self.encode_file(image)
            else:
                # PIL Image对象或numpy数组
                image_data = self.encode_image_object(image)
            
            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_data},
                        {"text": prompt}
                    ]
                }
            ]
            
            # 发送请求
            logger.info(f"发送图像编辑请求 | 提示: {prompt}")
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=model,
                messages=messages,
                stream=False,
                watermark=False,
                negative_prompt=negative_prompt
            )
            
            # 处理响应
            if response.status_code == 200:
                logger.info("图像编辑成功")
                result = {
                    "status": "success",
                    "status_code": response.status_code,
                    "request_id": response.request_id,
                    "result_image_url": response.output.choices[0].message.content[0]['image'],
                    "usage": response.usage,
                    "model": model
                }
                return result
            else:
                logger.error(f"图像编辑失败 | 状态码: {response.status_code} | 错误: {response.message}")
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "error_code": response.code,
                    "error_message": response.message,
                    "model": model
                }
                
        except Exception as e:
            logger.error(f"图像编辑异常: {str(e)}")
            return {
                "status": "error",
                "error": f"编辑失败: {str(e)}",
                "model": model
            }

    def download_image_from_url(self, url, save_path):
        """
        从URL下载图像并保存到本地
        
        Args:
            url: 图像URL
            save_path: 保存路径
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
            logger.info(f"图像已保存到: {save_path}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"下载图像失败: {str(e)}")
            return False

    def batch_edit_images(self, images_dir, prompts, save_dir, negative_prompt="", model="qwen-image-edit"):
        """
        批量编辑图像
        
        Args:
            images_dir: 图像文件夹路径
            prompts: 多组提示词，实际使用的时候随机挑一个
            negative_prompt: 负面提示词（可选）
            model: 模型名称
            
        Returns:
            list: 编辑结果列表
        """
        os.makedirs(save_dir, exist_ok=True)
        images = os.listdir(images_dir)
        np.random.shuffle(images)
        images = images[:3]
        results = []
        
        for i, image_file in enumerate(images):
            logger.info(f"处理第 {i+1}/{len(images)} 张图像")
            cur_image_path = os.path.join(images_dir, image_file)
            cur_prompt = prompts[i % len(prompts)]
            cur_image = Image.open(cur_image_path).convert("RGB")
            cur_image = np.array(cur_image)
            cur_image = np.ascontiguousarray(cur_image[:cur_image.shape[0]//2])
            cur_image = Image.fromarray(cur_image)
            result = self.edit_image(cur_image, cur_prompt, negative_prompt, model)
            result["batch_index"] = i
            results.append(result)
            save_path = os.path.join(save_dir, image_file)
            if result["status"] == "success":
                # 使用辅助方法下载图像
                self.download_image_from_url(result["result_image_url"], save_path)
            else:
                logger.error(f"图像编辑失败 | 状态码: {result['status_code']} | 错误: {result['error_message']}")
        
        return results

# 使用示例
if __name__ == "__main__":
    # 创建图像编辑器实例
    editor = QwenImageEdit(api_key="sk-0573a9a3207e4c7ba6df9ff22861e4fe")
    
    # 示例1：使用文件路径编辑图像
    # result = editor.edit_image(
    #     image="./road.png",
    #     prompt="在图中的路面上添加两个小树枝，稍微干枯的感觉，没有叶子，但是会影响行车，一定不要改变背景和整个图像的构图"
    # )
    # print("编辑结果:", result)
    
    # 示例2：使用PIL Image对象编辑图像
    # from PIL import Image
    # image = Image.open("your_image.jpg")
    # result = editor.edit_image(
    #     image=image,
    #     prompt="将背景改为海滩场景"
    # )
    # print("编辑结果:", result)
    
    # 示例3：批量编辑
    images_dir = "/perception/third_party/sixents/20250918/rosbag2_2025_09_18-06_10_27.parsed/camera_front"
    prompts = ["在图中的路面上添加两个小树枝，稍微干枯的感觉，没有叶子，一定不要改变背景和整个图像的构图",
               "在图中的路面上添加一个树枝，稍微干枯的感觉，没有叶子，但是会影响行车，一定不要改变背景和整个图像的构图",
               "在图中的路面上添加两个小树枝，带有绿色的叶子，但是会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加一个树枝，带有绿色的叶子，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上稍微远一点，添加一个树枝，带有绿色的叶子，稍微会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上稍微远一点，添加两个小树枝，稍微干枯的感觉，没有叶子，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加三个小树枝，稍微干枯的感觉，没有叶子，但是会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加一个较粗的树枝，带有绿色的叶子，明显会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上稍微近一点，添加一个树枝，稍微干枯的感觉，没有叶子，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加两个小树枝，一个干枯无叶，一个带有绿色叶子，会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面中央位置添加一个树枝，稍微干枯的感觉，没有叶子，明显会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加三个小树枝，带有绿色的叶子，稍微会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面左侧添加一个树枝，稍微干枯的感觉，没有叶子，一定不要改变背景和整个图像的构图",
            #    "在图中的路面右侧添加两个小树枝，带有绿色的叶子，但是会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上稍微远一点，添加一个较粗的树枝，稍微干枯的感觉，没有叶子，稍微会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加一个树枝，带有少量枯黄的叶子，但是会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加两个小树枝，一个在近处一个在远处，稍微干枯的感觉，没有叶子，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加一个弯曲的树枝，带有绿色的叶子，明显会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上添加两个小树枝，呈交叉状，稍微干枯的感觉，没有叶子，会影响行车，一定不要改变背景和整个图像的构图",
            #    "在图中的路面上稍微近一点，添加三个小树枝，带有绿色的叶子，但是会影响行车，一定不要改变背景和整个图像的构图"
               ]
    save_dir = "./results"
    results = editor.batch_edit_images(images_dir, prompts, save_dir)
    print("批量编辑结果:", results)
