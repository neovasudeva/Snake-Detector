# base python image
FROM python:3.6-alpine

# dependencies necessary to install numpy
RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev \
    libffi-dev openssl-dev \
    zlib-dev libjpeg \
    jpeg-dev

# create workdir and copy source code
WORKDIR /frontend
COPY . /frontend

# install dependencies in virtual env 
RUN python -m venv env \
    && env/bin/pip3 install --upgrade pip \
    && env/bin/pip3 install -r requirements.txt

# add virtual env bin to path
ENV PATH env/bin:$PATH

# start virtual env
RUN source env/bin/activate

# expose ports and run
EXPOSE 5000
ENTRYPOINT ["gunicorn"]
CMD ["-b", "0.0.0.0:5000", "-w", "4", "--access-logfile", "-", "--error-logfile", "-", "wsgi"]

