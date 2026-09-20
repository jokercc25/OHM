FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    TZ=Asia/Shanghai

# tzdata so TZ=Asia/Shanghai resolves to the correct local time
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the project. Secrets under .ohmo/ and .openharness/ are excluded
# by .dockerignore and mounted as volumes at runtime instead.
COPY . .

# Install the package plus the QQ bot SDK (qq-botpy is not in pyproject.toml)
RUN pip install . "qq-botpy"

# Config / state locations (mounted as volumes at runtime — see docker-compose.yml)
ENV OHMO_WORKSPACE=/workspace \
    OPENHARNESS_CONFIG_DIR=/config

CMD ["ohmo", "gateway", "run", "--workspace", "/workspace"]
