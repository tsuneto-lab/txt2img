FROM nvidia/cuda:11.3.1-base-ubuntu20.04
# no 22.04 image for cuda 11.3

# avoid being asked for timezone during apt install
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y --no-install-recommends libopencv-dev git python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Install remote packages
COPY ./requirements-remote.txt /app/requirements-remote.txt
WORKDIR /app
RUN pip3 install -r requirements-remote.txt

# Install local package (dependencies of itself)
COPY setup.py requirements-local.txt requirements.txt /app/
RUN pip3 install -r requirements.txt

# cache transformer pretrained models
COPY txt2img/configs /app/txt2img/configs
COPY scripts/txt2img_prefetch.py /app/scripts/txt2img_prefetch.py
RUN python3 scripts/txt2img_prefetch.py

# copy rest of the project and install (so local change would affect only this layer)
COPY . /app
