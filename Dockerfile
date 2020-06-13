# base image
FROM ubuntu:18.04

# install python, pip, and dependencies
RUN apt-get update \
  && apt-get install -y libsm6 libxext6 libxrender-dev \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install -y git 

# copy over source code
WORKDIR /app
COPY . /app

# install dependencies
RUN pip3 install -r requirements.txt \
  && pip3 install pycocotools \
  && pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
  && pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# expose ports 
EXPOSE 5000

# run index.py
ENTRYPOINT ["python3"]
CMD ["index.py"]
