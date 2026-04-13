FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y libgdal-dev g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "predict.py"]