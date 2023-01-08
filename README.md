# txt2img
txt2img module for stable-diffusion

## How to start

```bash
pip install git+https://github.com/tsuneto-lab/txt2img.git@main#egg=txt2img
```

Also, you can use examples/* files to start your project.

```bash
cp -r examples/* $PATH_TO_YOUR_PROJECT
cd $PATH_TO_YOUR_PROJECT

docker build . -t dev
docker run dev
```

As a reference, please look into scripts/txt2img_process.py as a reference.  

## txt2img_process script

You can start with trying out txt2img_process.py script in docker.  
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
  --prompt "a photograph of an astronaut riding a horse" \
  --neg_prompt ""

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

Try out changes without rebuilding the image:

```bash
docker run --rm --gpus all \
  -v $(pwd)/txt2img:/app/txt2img -v $(pwd)/scripts:/app/scripts \
  -v $HOME/outputs:/app/outputs -v $HOME/models:/app/models/ldm/stable-diffusion-v1 \
  tsuneto-lab/txt2img python3 scripts/txt2img_process.py --n_samples 1 \
  --ckpt /app/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt \
  --prompt "a photograph of an astronaut riding a horse"
```
