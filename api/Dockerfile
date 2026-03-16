# 1.确认基础镜像源
FROM python:3.12-slim

# 2.配置容器工作空间
WORKDIR /app

# 额外的 Python 包索引，默认使用国内镜像以降低构建阶段超时概率
ARG PYTHON_PACKAGE_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_DEFAULT_INDEX=${PYTHON_PACKAGE_INDEX_URL}
ENV PIP_INDEX_URL=${PYTHON_PACKAGE_INDEX_URL}

# 3.安装curl + docker-cli
# 仅保留 Docker CLI，避免把 daemon/containerd 一并装进镜像导致内存占用过高
RUN apt-get update && apt-get install -y --no-install-recommends curl docker-cli && \
    rm -rf /var/lib/apt/lists/*

# 4.安装v22版本的node.js
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# 5.安装uv, 使用uv来同步依赖环境
RUN pip install uv

# 6.拷贝requirements.txt并安装依赖
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# 7.拷贝项目文件(忽略开发环境的库)
COPY . .

# 8.为运行脚本配置权限
RUN chmod +x run.sh

# 9.设置Python路径变量
ENV PYTHONPATH=/app

# 10.暴露端口号
EXPOSE 8000

# 11.启动命令脚本
CMD ["./run.sh"]
