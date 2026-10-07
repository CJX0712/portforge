# PortForge · 世界级投资组合优化系统 — 作者：晨星 (CJX0712)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir pytest ruff

# 一键复现：跑 demo（含确定性校验）
CMD ["python", "examples/run_demo.py"]
