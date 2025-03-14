import base64
import glob
import os
import re
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Dict
import hashlib
import cv2
import fitz
from PIL import Image, ImageDraw, ImageFont
from doclayout_yolo import YOLOv10
from tqdm import tqdm
from zhipuai import ZhipuAI
import uuid
root_path="files"
model = YOLOv10("models/doclayout_yolo_ft.pt")
def truncate_pdf(pdf_path, n_pages):
    """
    将PDF文件截断为前n页并替换原文件
    返回最终文件名（即原文件名）

    参数：
    pdf_path - 原始PDF文件路径（字符串）
    n_pages  - 需要保留的页数（整数）

    返回：
    成功时返回原文件名，失败时抛出异常
    """
    # 参数校验
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    if n_pages <= 0:
        raise ValueError("保留页数必须大于0")

    # 生成临时文件名（同一目录）
    temp_path = os.path.join(
        os.path.dirname(pdf_path),
        f"temp_{uuid.uuid4().hex}_{os.path.basename(pdf_path)}"
    )

    try:
        # 第一步：创建临时文件
        with fitz.open(pdf_path) as src_doc:
            total_pages = src_doc.page_count
            if n_pages > total_pages:
                # raise ValueError(
                #     f"请求页数({n_pages})超过总页数({total_pages})")
                # n_pages==total_pages
                return pdf_path
            # 创建新文档并复制页面
            with fitz.open() as dst_doc:
                dst_doc.insert_pdf(src_doc, from_page=0, to_page=n_pages - 1)
                dst_doc.save(
                            temp_path,
                            deflate=True,  # 启用压缩
                            garbage=3,  # 清理冗余数据
                            clean=True,  # 优化文件结构
                            deflate_images=True,  # 图像压缩
                            deflate_fonts=True  # 字体压缩
                             )

        # 第二步：验证临时文件
        with fitz.open(temp_path) as temp_doc:
            if temp_doc.page_count != n_pages:
                raise RuntimeError("临时文件页数验证失败")

        # 第三步：删除原文件
        # os.remove(pdf_path)

        # 第四步：重命名临时文件
        # os.rename(temp_path, pdf_path)

        return temp_path

    except Exception as e:
        # 异常处理：清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(f"清理临时文件失败: {cleanup_error}")
        raise RuntimeError(f"PDF处理失败: {str(e)}") from e
def generate_folder_name(file_path,api_key, algorithm='sha256'):
    """
    通过文件内容生成唯一的哈希文件夹名
    :param file_path: 要处理的文件路径
    :param algorithm: 哈希算法（默认sha256）
    :return: 16进制哈希字符串
    """
    # 创建哈希对象
    hasher = hashlib.new(algorithm)

    try:
        with open(file_path, 'rb') as f:
            # 分块读取文件（处理大文件）
            while chunk := f.read(4096):
                hasher.update(chunk)
    except FileNotFoundError:
        raise ValueError(f"文件不存在: {file_path}")
    except IsADirectoryError:
        raise ValueError(f"路径是目录而不是文件: {file_path}")

    return os.path.join(root_path,api_key.split(".")[1], hasher.hexdigest())

def dir_init(output_dir):
    try:
        shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)
    except:
        os.makedirs(output_dir, exist_ok=True)


def calculate_font_size(text, target_coords, font_path, margin=5):
    # 解包目标区域坐标
    x1, y1, x2, y2 = target_coords

    # 计算可用尺寸
    total_width = (x2 - x1) - 2 * margin
    total_height = (y2 - y1) - 2 * margin

    # 扩展初始字体范围并动态计算上限
    low, high = 1, 50  # 初始上限50
    best_size = 0
    best_lines = []

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, mid)

        # 获取字体度量
        ascent, descent = font.getmetrics()
        line_height = ascent + descent

        # 改进行数计算逻辑
        max_lines = total_height // line_height

        # 特殊处理单行情况：当行高超过但ascent仍在可用高度内时
        if max_lines == 0:
            if ascent <= total_height:
                max_lines = 1  # 强制允许一行显示
            else:
                high = mid - 1
                continue

        # 文本分割逻辑
        lines = []
        current_line = []
        current_width = 0
        line_count = 0

        for char in text:
            char_width = font.getlength(char)
            if current_width + char_width <= total_width:
                current_line.append(char)
                current_width += char_width
            else:
                # 换行处理
                lines.append(''.join(current_line))
                line_count += 1
                current_line = [char]
                current_width = char_width

                # 行数检查
                if line_count >= max_lines:
                    break

        # 添加最后一行
        if current_line and line_count < max_lines:
            lines.append(''.join(current_line))
            line_count += 1

        # 验证文本完整性
        processed_length = sum(len(line) for line in lines)
        if processed_length == len(text):
            # 成功容纳，尝试更大字体
            if mid > best_size:  # 确保记录最大可用字号
                best_size = mid
                best_lines = lines
            low = mid + 1
        else:
            # 容纳失败，减小字体
            high = mid - 1

    return best_size, best_lines


def replace_text_in_image(
        image_path,
        output_path,
        target_coords,  # (x1, y1, x2, y2)
        text,
        font_path="msyh.ttc",  # 微软雅黑字体（需确保存在）
        margin=2,
        text_color=(0, 0, 0),  # 默认黑色
        bg_color=(255, 255, 255)  # 默认白色背景
):
    if text==None:
        return output_path
    x1, y1, x2, y2=target_coords
    font_size, lines = calculate_font_size(text, target_coords, font_path, margin)
    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    # 打开原始图片
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    # 擦除原始区域
    draw.rectangle(target_coords, fill=bg_color)

    # 计算文本绘制起始位置（垂直居中）
    text_y = y1 + margin

    # 逐行绘制文本
    for line in lines:
        text_x = x1 + margin  # 水平居中
        draw.text((text_x, text_y), line, font=font, fill=text_color)
        text_y += line_height

    img.save(output_path)

    return output_path
def glm_ocr_translate(image_path,api_key):
    client = ZhipuAI(api_key=api_key)
    with open(image_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
                    你正在处理学术论文图像，请严格按以下步骤执行：
                    1. OCR识别：完整精确提取图片中的文本，保留公式和特殊符号
                    2. 学术翻译：将内容翻译为中文，遵循以下规则：
                       - Introduction → 引言
                       - Conclusion → 结论
                       - Theorem → 定理
                       - 保留LaTeX公式
                       - 学术术语优先使用《英汉学术翻译规范》标准译法
                    3. 输出格式：仅返回JSON结构：
                       { "translated_text": "..."}
                    """},
                    {"type": "image_url", "image_url": {"url": img_base}}
                ]
            }
        ],
        top_p=0.7,
        temperature=0.95,
        # max_tokens=1024,
    )
    result=response.choices[0].message.content
    # print("json处理之前",result)
    # print("#"*100)
    result = extract_json_from_markdown(result)
    return result

def glm_batch_translate(req_lists, api_key):
    """带进度条的并行翻译函数

    参数：
    - req_lists: 需要翻译的文本列表
    - api_key: API密钥

    返回：
    - 按原始顺序排列的翻译结果列表
    """
    # 准备翻译任务
    items_to_translate = {i: text for i, text in enumerate(req_lists)}
    total_items = len(items_to_translate)

    # 初始化线程安全组件
    lock = threading.Lock()
    progress_bar = tqdm(total=total_items, desc="翻译进度", unit="item")
    translated_items = {}
    error_log = []

    def _safe_update_progress():
        """线程安全的进度更新"""
        with lock:
            progress_bar.update(1)

    def _log_error(msg):
        """线程安全的错误日志记录"""
        with lock:
            error_log.append(msg)
            progress_bar.write(msg)

    # 使用上下文管理器管理线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有翻译任务
        future_map = {
            executor.submit(glm_ocr_translate, text, api_key): idx
            for idx, text in items_to_translate.items()
        }

        # 实时处理完成的任务
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                result = future.result()
                translated_items[idx] = result
                _safe_update_progress()
            except Exception as e:
                error_msg = f"翻译失败 第{idx}项: {str(e)}"
                _log_error(error_msg)
                translated_items[idx] = None  # 保留位置

    # 关闭进度条
    progress_bar.close()

    # 输出错误汇总
    if error_log:
        print("\n错误汇总：")
        for error in error_log:
            print(f"• {error}")

    # 按原始顺序返回结果
    return [translated_items[i] for i in sorted(translated_items)]
def validate_api_key(image_path,api_key):
    try:
        client = ZhipuAI(api_key=api_key)
        with open(image_path, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')
        response = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
                            你正在处理学术论文图像，请严格按以下步骤执行：
                            1. OCR识别：完整精确提取图片中的文本，保留公式和特殊符号
                            2. 学术翻译：将内容翻译为中文，遵循以下规则：
                               - Introduction → 引言
                               - Conclusion → 结论
                               - Theorem → 定理
                               - 保留LaTeX公式
                               - 学术术语优先使用《英汉学术翻译规范》标准译法
                            3. 输出格式：仅返回JSON结构：
                               { "translated_text": "..."}
                            """},
                        {"type": "image_url", "image_url": {"url": img_base}}
                    ]
                }
            ],
            top_p=0.7,
            temperature=0.95,
            # max_tokens=1024,
        )
        result = response.choices[0].message.content
        return result,True
    except Exception as ex:
        return ex,False


def extract_json_from_markdown(text: str) -> Dict:
    # 专注于匹配 translated_text 的正则表达式
    pattern = r'"translated_text":\s*(["\'])(.*?)\1'
    matches = re.finditer(pattern, text, re.DOTALL)

    translated = []

    for match in matches:
        translated_text = match.group(2).replace('\n', ' ').strip()
        translated.append(translated_text)

    if translated:
        return {
            "translated_text": '\n'.join(translated)
        }
    else:
        return {
            "translated_text": None
        }

def images_to_pdf(image_folder, output_pdf="output.pdf", dpi=300):
    doc = fitz.open()
    for img_path in sorted(glob.glob(f"{image_folder}/*")):
        # 创建与图片尺寸一致的页面
        img = fitz.open(img_path)
        rect = fitz.Rect(0, 0, img[0].rect.width, img[0].rect.height)
        page = doc.new_page(width=rect.width, height=rect.height)

        # 插入图片（支持调整位置和缩放）
        page.insert_image(rect, filename=img_path, xref=0)
        img.close()

    doc.save(output_pdf)
def pdf_to_images(pdf_path, output_folder):
    """PDF转高清图片（保持原始色彩）"""
    page_paths=[]
    try:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    except:
        os.makedirs(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300, colorspace="rgb")
        img_path=f"{output_folder}/page_{page_num + 1}.png"
        pix.save(img_path)
        page_paths.append(img_path)
    doc.close()
    return page_paths

def detect_text_regions(image_path, model):
    det_res = model.predict(
        image_path,
        imgsz=1024,
        conf=0.5,
        device="cuda:0"
    )

    regions = []
    for box in det_res[0].boxes:
        # names: {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table',
        #         6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}

        if box.cls in [0,1,4,6,7]: #只翻译1 title 2 plain text 4 figure_caption 6 table_caption 7 table_footnote
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            regions.append((x1, y1, x2, y2))
    return regions


def parallel_process_by_pagepath(all_crop_info, batch_results, max_workers=None):
    """
    基于page_path分组的并行处理函数

    参数：
    - all_crop_info: 包含图片信息的字典列表，每个字典必须包含page_path和coords
    - batch_results: 对应处理结果列表，每个元素应包含translated_text
    - max_workers: 最大线程数（默认自动计算为CPU核心数*2，上限32）
    """
    # 参数校验
    if len(all_crop_info) != len(batch_results):
        raise ValueError("all_crop_info和batch_results长度必须一致")

    # 初始化线程池配置
    final_max_workers = max_workers or min(os.cpu_count() * 2, 32)

    # 分组任务（自动合并相同page_path的任务）
    grouped_tasks = defaultdict(list)
    for info, result in zip(all_crop_info, batch_results):
        if "page_path" not in info or "coords" not in info:
            raise ValueError("all_crop_info元素必须包含page_path和coords字段")
        grouped_tasks[info["page_path"]].append((info, result))

    # 创建线程安全组件
    lock = threading.Lock()
    progress_bar = tqdm(total=len(all_crop_info), desc="文本替换")  # 移除了lock参数

    def _process_single_task(info, result):
        """处理单个任务的核心逻辑"""
        # 结果有效性检查
        if not result or "translated_text" not in result:
            with lock:  # 使用锁保护输出
                progress_bar.write(f"跳过处理: {info['page_path']} - 缺少翻译结果")
            return False

        # 执行文本替换
        try:
            replace_text_in_image(
                info["page_path"],
                info["page_path"],
                info["coords"],
                result["translated_text"]
            )
            return True
        except Exception as e:
            with lock:  # 使用锁保护输出
                progress_bar.write(f"处理失败: {info['page_path']} - {str(e)}")
            return False

    def _process_group(tasks):
        """处理单个page_path分组"""
        for info, result in tasks:
            _process_single_task(info, result)
            with lock:  # 使用锁保护进度更新
                progress_bar.update(1)

    # 执行并行处理
    with ThreadPoolExecutor(max_workers=final_max_workers) as executor:
        # 提交分组任务到线程池
        futures = [
            executor.submit(_process_group, tasks)
            for tasks in grouped_tasks.values()
        ]

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                with lock:
                    progress_bar.write(f"线程执行异常: {str(e)}")

    progress_bar.close()
    return True


def process_pdf(input_pdf, api_key,pages):
    """主处理流程（批量优化版）"""
    # output_pdf = input_pdf[:-4]+"_en2zh.pdf"
    input_pdf=truncate_pdf(input_pdf,pages)
    output_path = os.path.join(os.path.dirname(input_pdf), f"translated_{uuid.uuid4()}.pdf")

    img_folder = generate_folder_name(input_pdf,api_key)
    dir_init(img_folder)

    # 转换PDF为图片
    page_paths=pdf_to_images(input_pdf, img_folder)


    # 第一阶段：收集所有检测区域和裁剪图片
    all_crop_info = []
    for page_path in tqdm(page_paths, desc="检测文本区域"):
        img = cv2.imread(page_path)
        regions = detect_text_regions(page_path, model)

        # 生成唯一crop文件名（添加页面序号防冲突）
        page_num = int(re.search(r'page_(\d+)\.png', page_path).group(1))
        for idx, (x1, y1, x2, y2) in enumerate(regions):
            crop_path = f"{img_folder}/crop_p{page_num:03d}_{idx:03d}.png"
            cv2.imwrite(crop_path, img[y1:y2, x1:x2])
            all_crop_info.append({
                "page_path": page_path,
                "coords": (x1, y1, x2, y2),
                "crop_path": crop_path
            })

    # 第二阶段：批量处理所有裁剪图片
    if all_crop_info:
        crop_paths = [info["crop_path"] for info in all_crop_info]
        try:
            # 批量获取OCR结果
            batch_results = glm_batch_translate(crop_paths, api_key)
        except Exception as e:
            print(f"批量处理失败: {str(e)}")
            batch_results = [None] * len(crop_paths)

        parallel_process_by_pagepath(
            all_crop_info=all_crop_info,
            batch_results=batch_results,
            max_workers=8  # 可选参数
        )

        # 批量替换文本
        # for info, result in tqdm(zip(all_crop_info, batch_results),
        #                          desc="文本替换", total=len(all_crop_info)):
        #     if not result or "translated_text" not in result:
        #         print("获取结果失败")
        #         continue
        #
        #     try:
        #         replace_text_in_image(
        #             info["page_path"],
        #             info["page_path"],
        #             info["coords"],
        #             result["translated_text"]
        #         )
        #     except Exception as e:
        #         print(f"文本替换失败: {str(e)}")



    # 第三阶段：合成PDF
    doc = fitz.open()
    for img_path in tqdm(page_paths, desc="生成PDF"):
        # 自动获取图片尺寸
        # print("合并顺序",img_path)
        with fitz.open(img_path) as img:
            page = doc.new_page(width=img[0].rect.width,
                                height=img[0].rect.height)
            page.insert_image(page.rect, filename=img_path)
    doc.save(
        output_path,
        deflate=True,  # 启用压缩
        garbage=3,  # 清理冗余数据
        clean=True,  # 优化文件结构
        deflate_images=True,  # 图像压缩
        deflate_fonts=True  # 字体压缩
    )
    return output_path

    # 清理临时文件（按需启用）
    # shutil.rmtree(img_folder)