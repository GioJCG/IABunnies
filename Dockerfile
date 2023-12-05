FROM ubuntu:20.04

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y python3.8 python3.8-dev python3-pip libgl1-mesa-glx libglib2.0-0 \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=3000", "--reload"]
