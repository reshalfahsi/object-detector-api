FROM python:3.9

WORKDIR /src

COPY requirements.txt .

RUN \
  apt-get update -q -y && \
  apt-get clean && \
  rm -rf /var/cache/apt/* && \
  rm -rf /var/lib/apt/lists/* && \
  pip install pip --upgrade && \
  pip install -r requirements.txt --no-cache-dir


COPY translator translator
COPY main.py .

EXPOSE 80

ENTRYPOINT ["python3", "main.py"]
