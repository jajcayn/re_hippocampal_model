FROM ubuntu:latest
LABEL org.opencontainers.image.source=https://github.com/jajcayn/re_hippocampal_model

RUN apt update && apt -y upgrade
RUN apt install -y python3 python3-dev python3-pip

RUN mkdir re_hippocampus_model
WORKDIR re_hippocampus_model/
COPY . .

RUN pip3 install --upgrade -r requirements.txt

EXPOSE 8899
CMD ["jupyter", "lab", "--port=8899", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
