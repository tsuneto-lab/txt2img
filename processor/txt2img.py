import torch


class Txt2imgProcessor:
    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler

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
        return x_samples_ddim
