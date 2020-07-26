# base image
FROM python:3.6-slim

# create workdir and copy source code
WORKDIR /backend
COPY . /backend

# install build dependencies, change permissions for wait script, install dependencies, clean up
RUN apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 --no-install-recommends \
    && apt-get install -y g++ gcc python3-pip python3-dev --no-install-recommends \ 
    && apt-get install -y git \
    && chmod 755 /backend/wait-for-it.sh \
    && pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install 'git+https://github.com/facebookresearch/detectron2.git' \
    && apt remove -y gcc git python3-dev \ 
    && rm -rf /var/lib/apt/lists/*  

# add virtual env bin to path
ENV PATH env/bin:$PATH

# BUG FIX FOR MEM LEAK
ENV LRU_CACHE_CAPACITY 1

# expose ports and run
EXPOSE 5001
CMD ["gunicorn", "-b", "0.0.0.0:5001", "-w", "4", "--access-logfile", "-", "--error-logfile", "-", "wsgi"]
