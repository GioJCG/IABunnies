FROM ubuntu:20.04

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python3.8=3.8.10-0ubuntu1~20.04.1 python3.8-dev=3.8.10-0ubuntu1~20.04.1 python3.8-distutils=3.8.10-0ubuntu1~20.04.1 \
    && apt-get install -y python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=3000", "--reload"]
