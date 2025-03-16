FROM ubuntu:22.04
#使用cpu推理用ubuntu:22.04 GPU用nvidia/cuda:12.1.0-base-ubuntu22.04
#FROM nvidia/cuda:12.1.0-base-ubuntu22.04
# 设置清华源加速 apt 和 pip
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
#
## 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app
# 将当前目录下的所有文件复制到工作目录
COPY ./requirements.txt /app/requirements.txt
COPY ./app.py /app/app.py
COPY ./utils.py /app/utils.py
COPY ./main.py /app/main.py
COPY ./fonts/ /app/fonts
COPY ./models/ /app/models
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 暴露端口
EXPOSE 9000

CMD ["python3", "app.py"]