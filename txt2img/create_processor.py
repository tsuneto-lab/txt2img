import importlib.resources as pkg_resources

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from omegaconf import OmegaConf
from imwatermark import WatermarkEncoder

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from .processor import Txt2imgProcessor
from .load_model import load_model_from_config
from . import configs


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
    safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def create_processor(ckpt: str, sampler: str, precision: str, nsfw_replacement: str):
    config_txt = pkg_resources.read_text(configs, "sd-v1-inference.yaml")
    config = OmegaConf.create(config_txt)
    model = load_model_from_config(config, ckpt)

    if sampler == "dpm_solver":
        sampler = DPMSolverSampler(model)
    elif sampler == "plms":
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    processor = Txt2imgProcessor(
        model, sampler, safety_feature_extractor, safety_checker, create_wm_encoder(), precision, nsfw_replacement)

    return processor


def create_wm_encoder():
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    return wm_encoder
