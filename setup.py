from setuptools import setup, find_packages

setup(
    name='txt2img',
    version='0.0.1',
    description='',
    packages=find_packages(),
    package_data={'txt2img.configs': ['*.yml', '*.yaml']},
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "diffusers",
        "Pillow",
        "einops",
        "invisible-watermark",
        "omegaconf",
        "transformers",
        "pytorch-lightning",
        # install a fork of stable-diffusion (for better package discovery)
        "latent-diffusion @ git+https://github.com/tsuneto-lab/stable-diffusion.git@33b08bff#egg=latent-diffusion",
        # below required by stable diffusion
        "torchmetrics==0.6.0",
        "kornia",
        "taming-transformers-rom1504",
        "clip @ git+https://github.com/openai/CLIP.git@main#egg=clip"
    ],
)
