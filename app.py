# -*- coding: utf-8 -*-
import os
import sys
import threading
from contextlib import contextmanager

import gradio as gr
os.system("pip install gradio_pdf -U")
from gradio_pdf import PDF  # 严格使用gradio_pdf组件
from tqdm import tqdm

from utils import process_pdf, validate_api_key,convert_to_pdf_if_needed


# --------------------- 严格保持您提供的tqdm捕获器代码 ---------------------
class TQDMCapture:
    def __init__(self):
        self.buffer = []
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.buffer.append(text.strip())
            sys.__stdout__.write(text)  # 终端同步输出

    def flush(self):
        pass

    def get_full_output(self):
        with self.lock:
            return "\n".join(self.buffer)


@contextmanager
def capture_tqdm_output(capture_obj):
    original_init = tqdm.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["file"] = capture_obj
        original_init(self, *args, **kwargs)

    tqdm.__init__ = patched_init
    yield
    tqdm.__init__ = original_init


# --------------------- 修正后的处理函数 ---------------------
def get_info(input_pdf, api_key, pages):
    capture = TQDMCapture()
    result_path = None  # 存储最终结果路径

    def worker():
        nonlocal result_path
        with capture_tqdm_output(capture):
            try:
                # 文件预处理
                actual_path = convert_to_pdf_if_needed(input_pdf.name)
                capture.write(f"\n{os.path.basename(input_pdf.name)}文件开始处理")

                env_api_key = os.environ.get("api_key")
                if env_api_key:
                    api_key = env_api_key
                    env_api_result, env_api_status = validate_api_key("assets/img.png", env_api_key)
                    if not env_api_status:
                        raise gr.Error("环境变量中的api_key不正确", env_api_result)
                if not api_key:
                    raise gr.Error("api_key不能为空")
                api_result, api_status = validate_api_key("assets/img.png", api_key)
                if not api_status:
                    raise gr.Error(api_result)

                # 处理PDF
                result_path = process_pdf(actual_path, api_key, pages)
                capture.write(f"\n{os.path.basename(result_path)}文件翻译完成")
                capture.write("\n✅ 任务完成！")
            except Exception as e:
                capture.write(f"\n❌ 系统错误: {str(e)}")
                raise

    thread = threading.Thread(target=worker)
    thread.start()

    prev_output = ""
    while thread.is_alive():
        thread.join(timeout=0.2)
        current_output = capture.get_full_output()
        if current_output != prev_output:
            prev_output = current_output
            yield current_output, None, None  # 实时更新时保持空路径

    # 最终返回结果
    yield capture.get_full_output(), result_path, result_path


# --------------------- 严格保持原有界面结构 ---------------------
with gr.Blocks(title="PDF翻译工具",
               css=".download-box {border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px;}") as demo:
    # ================= 标题区域 =================
    gr.Markdown("# chaodreaming开源PDF翻译工具（英语转中文）使用glm-4v-flash，免费并且10QPS,速度约为10s/页")
    gr.Markdown("# api_key注册：https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys")
    gr.Markdown("# 开源地址：https://github.com/chaodreaming/layover-pdf")
    gr.Markdown("# modelscope测试：https://www.modelscope.cn/studios/chaodreaming/layover-pdf/summary")


    # ================= 控制区域 =================
    with gr.Row(variant="panel"):
        with gr.Column(scale=4, min_width=600):
            with gr.Group():
                file_input = gr.File(
                    label="📁 上传源文件",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],  # 支持图片格式
                    height=400
                )
                api_key_input = gr.Textbox(
                    label="🔑 API密钥",
                    type="password",
                    placeholder="请输入您的API密钥",
                    visible=not os.environ.get("api_key")
                )
                # 严格保持原有Slider组件
                page_slider = gr.Slider(1, 100,
                                        value=10,
                                        label="📄 转换页数范围",
                                        step=1,
                                        info="拖动选择需要转换的页码范围")

            with gr.Row():
                convert_btn = gr.Button("🔄 开始转换", variant="primary")
                clear_btn = gr.ClearButton(value="🧹 清除输入")
    # ================= 新增日志显示框 =================
    console_output = gr.Textbox(label="处理日志", lines=5,max_lines=5, interactive=False,autoscroll=True)

    # ================= 内容区域 =================
    with gr.Row():
        with gr.Column(scale=4, min_width=500):
            gr.Markdown("### 源文件预览")
            original_view = PDF(label="", height=600)  # 使用gradio_pdf.PDF

        with gr.Column(scale=6):
            with gr.Column(elem_classes="download-box"):
                gr.Markdown("### 翻译结果下载")
                result_file = gr.File(
                    label="⬇️ 点击下载翻译后的文件",
                    interactive=False,
                    height=100
                )

            with gr.Column():
                gr.Markdown("### 翻译结果预览")
                translated_view = PDF(label="", height=500)  # 使用gradio_pdf.PDF

    # ================= 事件绑定 =================
    file_input.change(
        fn=lambda f: convert_to_pdf_if_needed(f.name),
        inputs=file_input,
        outputs=original_view
    )

    convert_btn.click(
        fn=get_info,
        inputs=[file_input, api_key_input, page_slider],
        outputs=[console_output, result_file, translated_view]  # 输出到三个组件
    )

    clear_btn.add([
        file_input, api_key_input, page_slider,
        original_view, translated_view, result_file, console_output
    ])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9000)
