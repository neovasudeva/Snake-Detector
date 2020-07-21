# base image
FROM python:3.6-stretch

# install requirements
RUN apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && apt-get install -y gcc python3-pip python3-dev 

# create workdir and copy source code
WORKDIR /backend
COPY . /backend

# change permissions on wait-for-it script
RUN chmod 755 /backend/wait-for-it.sh

# install dependencies in virtual env 
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# add virtual env bin to path
ENV PATH env/bin:$PATH

# expose ports and run
EXPOSE 5001
CMD ["gunicorn", "-b", "0.0.0.0:5001", "--access-logfile", "-", "--error-logfile", "-", "wsgi"]
