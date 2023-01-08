FROM nvidia/cuda:11.3.1-base-ubuntu20.04
# no 22.04 image for cuda 11.3

# avoid being asked about timezone during apt install
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y --no-install-recommends libopencv-dev git python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

# Install packages
RUN pip3 install -r requirements.txt

# cache transformer pretrained models
RUN python3 scripts/txt2img_prefetch.py
