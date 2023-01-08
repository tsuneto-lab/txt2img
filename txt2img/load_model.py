import torch
import importlib.resources as pkg_resources
from omegaconf import OmegaConf
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from ldm.util import instantiate_from_config

from . import configs


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    return model


def prefetch():
    # load safety model
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    AutoFeatureExtractor.from_pretrained(safety_model_id)
    StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    config_txt = pkg_resources.read_text(configs, "sd-v1-inference.yaml")
    config = OmegaConf.create(config_txt)
    instantiate_from_config(config.model)
