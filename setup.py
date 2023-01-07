from setuptools import setup, find_packages

setup(
    name='txt2img',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "Pillow",
        "einops",
        "invisible-watermark"
    ],
)
