FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir sentence-transformers==6.1.0 numpy==2.5.3 pandas==3.0.6 fastapi==0.141.1 uvicorn==0.53.0
COPY src src
COPY api api
ENV HF_HOME=/cache
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
