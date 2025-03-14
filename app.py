import gradio as gr
from gradio_pdf import PDF
import os
import uuid
import pymupdf
from utils import process_pdf,validate_api_key

def get_info(input_pdf, api_key, pages):
    if not api_key:
        raise gr.Error( "api_key不能为空")
    api_result,api_status=validate_api_key("models/img.png",api_key)
    if not api_status:
        raise gr.Error(api_result)

    try:
        output_path = process_pdf(input_pdf, api_key,pages)
    except Exception as ex:
        raise  gr.Error(str(ex))

    return output_path, output_path


def convert_to_pdf_if_needed(file_path):
    """图片转PDF处理"""
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        pdf_bytes = f.convert_to_pdf()
        tmp_path = os.path.join(os.path.dirname(file_path), f"{uuid.uuid4()}.pdf")
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)
        return tmp_path


with gr.Blocks(title="PDF翻译工具",
               css=".download-box {border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px;}") as demo:
    # ================= 标题区域 =================
    gr.Markdown("# PDF翻译工具（英语转中文）使用glm-4v-flash，免费并且10QPS,速度约为10s/页", elem_id="title")
    gr.Markdown("# api_key注册：https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys", elem_id="title")
    gr.Markdown("# 开源地址：https://github.com/chaodreaming/layover-pdf", elem_id="title")

    # ================= 控制区域 =================
    with gr.Row(variant="panel"):
        with gr.Column(scale=4, min_width=600):
            with gr.Group():
                file_input = gr.File(label="📁 上传源文件",
                                     file_types=[".pdf"],
                                     height=400)
                api_key_input = gr.Textbox(label="🔑 API密钥",
                                           type="password",
                                           placeholder="请输入您的API密钥")
                page_slider = gr.Slider(1, 100,
                                        value=10,
                                        label="📄 转换页数范围",
                                        step=1,
                                        info="拖动选择需要转换的页码范围")

            with gr.Row():
                convert_btn = gr.Button("🔄 开始转换", variant="primary")
                # 修正的清除按钮
                clear_btn = gr.ClearButton(value="🧹 清除输入")  # 正确用法

    # ================= 内容区域 =================
    with gr.Row():
        with gr.Column(scale=4, min_width=500):
            gr.Markdown("### 源文件预览")
            original_view = PDF(label="", height=600)

        with gr.Column(scale=6):
            with gr.Column(elem_classes="download-box"):
                gr.Markdown("### 翻译结果下载")
                result_file = gr.File(label="⬇️ 点击下载翻译后的文件",
                                      interactive=False,
                                      height=100)

            with gr.Column():
                gr.Markdown("### 翻译结果预览")
                translated_view = PDF(label="", height=500)

    # ================= 事件绑定 =================
    file_input.change(
        convert_to_pdf_if_needed,
        inputs=file_input,
        outputs=original_view
    )

    convert_btn.click(
        get_info,
        inputs=[file_input, api_key_input, page_slider],
        outputs=[result_file, translated_view]
    )

    # 修正的清除按钮绑定
    clear_btn.add(
        [file_input, api_key_input, page_slider, original_view, translated_view, result_file]
    )

if __name__ == "__main__":
    demo.launch()
