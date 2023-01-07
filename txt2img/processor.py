import torch
import numpy as np
from PIL import Image


class Txt2imgProcessor:
    def __init__(self, model, sampler, safety_feature_extractor, safety_checker):
        self.model = model
        self.sampler = sampler
        self.safety_feature_extractor = safety_feature_extractor
        self.safety_checker = safety_checker

    def process(self, scale: float, batch_size: int, prompts, channels: int,
                factor: int, height: int, width: int, ddim_steps: int,
                ddim_eta: float, x_T):
        uc = None
        if scale != 1.0:
            uc = self.model.get_learned_conditioning(
                batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        c = self.model.get_learned_conditioning(prompts)
        shape = [channels, height // factor, width // factor]
        samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                              conditioning=c,
                                              batch_size=batch_size,
                                              shape=shape,
                                              verbose=False,
                                              unconditional_guidance_scale=scale,
                                              unconditional_conditioning=uc,
                                              eta=ddim_eta,
                                              x_T=x_T)

        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp(
            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

        x_checked_image, has_nsfw_concept = self.check_safety(x_samples_ddim)
        x_checked_image_torch = torch.from_numpy(
            x_checked_image).permute(0, 3, 1, 2)

        return x_checked_image_torch

    def check_safety(self, x_image):
        safety_checker_input = self.safety_feature_extractor(
            self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(
            images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = self.load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def load_replacement(self, x):
        try:
            hwc = x.shape
            y = Image.open(
                "assets/cat.jpg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y)/255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x
