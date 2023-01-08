import torch
import cv2
import numpy as np
from einops import rearrange
from PIL import Image
from imwatermark import WatermarkEncoder
from torch import autocast
from contextlib import nullcontext


class Txt2imgProcessor:
    def __init__(self, model, sampler, safety_feature_extractor, safety_checker, precision: str):
        self.model = model
        self.sampler = sampler
        self.safety_feature_extractor = safety_feature_extractor
        self.safety_checker = safety_checker
        self.wm_encoder = self.init_wm_encoder()
        self.precision_scope = autocast if precision == "autocast" else nullcontext

    def init_wm_encoder(self):
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
        return wm_encoder

    def process(self, scale: float, batch_size: int, prompt: str, channels: int,
                factor: int, height: int, width: int, ddim_steps: int,
                ddim_eta: float, x_T: torch.Tensor):
        with torch.no_grad():
            with self.precision_scope("cuda"):
                uc = None
                if scale != 1.0:
                    uc = self.model.get_learned_conditioning(
                        batch_size * [""])
                c = self.model.get_learned_conditioning(batch_size * [prompt])
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

                x_checked_image, has_nsfw_concept = self.check_safety(
                    x_samples_ddim)
                x_checked_image_torch = torch.from_numpy(
                    x_checked_image).permute(0, 3, 1, 2)

                imgs = []
                for x_sample in x_checked_image_torch:
                    img = self.x_sample2img(x_sample)
                    img = self.put_watermark(img)
                    imgs.append(img)

                return imgs

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

    def x_sample2img(self, x_sample):
        x_sample = 255. * \
            rearrange(x_sample.cpu().numpy(),
                      'c h w -> h w c')
        return Image.fromarray(
            x_sample.astype(np.uint8))

    def put_watermark(self, img):
        if self.wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = self.wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img
