# layover-pdf 保留格式的PDF翻译

一个使用layout overlay方式实现PDF英文翻译为中文的免费工具，使用智普api来完成ocr和翻译功能

chaodreaming开源PDF翻译工具（英语转中文）使用glm-4v-flash

glm-4v-flash免费并且10QPS,实测速度约为10s/页

api_key注册：https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys

开源地址：https://github.com/chaodreaming/layover-pdf

modelscope测试：https://www.modelscope.cn/studios/chaodreaming/layover-pdf/summary

modelscope无法承受很大的并发，当无法使用时可以自行在本地构建使用app.py

### 效果演示

<div align="center">
<img src="./assets/preview.gif" width="80%"/>
</div>

## Quick CPU Demo


`git clone  https://github.com/chaodreaming/layover-pdf`

`cd layover-pdf`

`pip install -r requirements.txt`



### 启动gradio服务

`python3 app.py`

#### 服务地址：

`http://localhost:9000/`

### 命令行参数方式

```
python main.py --input_pdf pdf_pat/img_path --api_key xxx --pages 100
```

### docker使用方式

```
docker build -t layover-pdf .
```

```
docker run  -p 9000:9000 --gpus=all -it layover-pdf /bin/bash  
```

### 更新日志



2025.3.18 新增支持img和打印log进度

2025.3.16上线PDF保留布局翻译功能，目前支持英译中

### Development Guide

TODO

# License Information



[LICENSE.md](https://github.com/chaodreaming/layover-pdf/blob/main/LICENSE)

This project currently uses PyMuPDF to achieve advanced functionality. However, since it adheres to the AGPL license, it may impose restrictions on certain usage scenarios. In future iterations, we plan to explore and replace it with a more permissive PDF processing library to enhance user-friendliness and flexibility.

# Acknowledgments



- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

- [Noto Serif Simplified Chinese - Google Fonts](https://fonts.google.com/noto/specimen/Noto+Serif+SC)



# Star History

<a href="https://star-history.com/#chaodreaming/layover-pdf&Date">

 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=chaodreaming/layover-pdf&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=chaodreaming/layover-pdf&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=chaodreaming/layover-pdf&type=Date"/>
 </picture>

<a>