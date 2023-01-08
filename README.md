# txt2img
txt2img module for stable-diffusion

## Aim

- A simple txt2img-processor Class that wraps stable-diffusion and can be embedded into command line scripts, REST servers, queue consumers, twitter bots, etc.
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) as external dependency

## How to use

You can start with trying out txt2img.py script in docker.  
It requires nvidia-docker setup.  
FYI, ansible playbook for setting up nvidia-docker in WSL2: https://github.com/tsuneto-lab/wsl-setup

```bash
# clone this repo and cd into it
git clone https://github.com/tsuneto-lab/stable-diffusion.git
cd stable-diffusion

# build the image
docker build . -t tsuneto-lab/txt2img

# put a model (sd-v1-4.ckpt here) under $HOME/models/ and run
docker run --rm --gpus all \
  -v $HOME/outputs:/app/outputs -v $HOME/models:/app/models/ldm/stable-diffusion-v1 \
  tsuneto-lab/txt2img python3 scripts/txt2img_process.py --n_samples 1 \
  --ckpt /app/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt \
  --prompt "a photograph of an astronaut riding a horse"

# it would be convenient to access outputs dir from windows explorer (if wsl)
# as well as simply serve the directory:
# python3 -m http.server 8080 --bind 0.0.0.0 --directory $HOME/outputs
```

## Develop Environment

```bash
conda env create -f environment.yaml
conda activate txt2img-dev
# install required packages mainly for vscode integration such as autocomplete and popups
# if you don't need them, you can just build and run the docker container
```