# Initially copied from CompVis/stable-diffusion/scripts/txt2img.py
# Logics are being moved into local txt2img package

import argparse
import os

from txt2img.create_processor import create_processor
from txt2img.seed_everything import seed_everything


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        nargs="?",
        default="",
        help="the negative prompt"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="specify sampler: [ddim(default), plms, dpm_solver]",
        default="ddim"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    seed_everything(opt.seed)

    processor = create_processor(
        f"{opt.ckpt}", opt.sampler, opt.precision)

    sample_path = opt.outdir
    base_count = len(os.listdir(sample_path))

    imgs = processor.process(scale=opt.scale,
                             batch_size=opt.n_samples,
                             prompt=opt.prompt,
                             negative_prompt=opt.neg_prompt,
                             channels=opt.C,
                             factor=opt.f,
                             height=opt.H,
                             width=opt.W,
                             ddim_steps=opt.ddim_steps,
                             ddim_eta=opt.ddim_eta,
                             x_T=None)

    for img in imgs:
        img.save(os.path.join(
            sample_path, f"{base_count:05}.png"))
        base_count += 1

    print(f"Your samples are ready and waiting for you here: \n{sample_path} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
