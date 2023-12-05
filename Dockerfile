FROM ubuntu:20.04

RUN apt-get update -y 

RUN apt-get upgrade -y

RUN apt-get install python3.7.2 -y

RUN apt-get install python3-pip -y

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=3000", "--reload"]
