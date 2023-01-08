from omegaconf import OmegaConf
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from ldm.util import instantiate_from_config

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
AutoFeatureExtractor.from_pretrained(safety_model_id)
StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

config = OmegaConf.load("txt2img/configs/sd-v1-inference.yaml")
instantiate_from_config(config.model)
