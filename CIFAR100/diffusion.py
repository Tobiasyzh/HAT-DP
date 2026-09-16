import os

import torch
import torch.nn as nn
from torchvision.utils import save_image

from improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class Args:
    image_size = 32
    num_channels = 128
    num_res_blocks = 3
    num_heads = 4
    num_heads_upsample = -1
    attention_resolutions = "16,8"
    dropout = 0.3
    learn_sigma = True
    sigma_small = False
    class_cond = False
    diffusion_steps = 4000
    noise_schedule = "cosine"
    timestep_respacing = ""
    use_kl = False
    predict_xstart = False
    rescale_timesteps = True
    rescale_learned_sigmas = True
    use_checkpoint = False
    use_scale_shift_norm = True


class TemporaryGrad:
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


class DiffusionPurificationModel(nn.Module):
    def __init__(
            self,
            device,
            guide_type="osgd",
            checkpoint_path="cifar10_uncond_50M_500K.pt",
            **unused,
    ):
        super().__init__()
        if guide_type != "osgd":
            raise ValueError("This cleaned version only supports the original OSGD guidance.")

        self.device = device
        self.checkpoint_path = checkpoint_path

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

        self.model = model
        self.model.requires_grad_(False)
        self.diffusion = diffusion
        self.mse_loss = nn.MSELoss(reduction="none")

    def guide(self, x_t, reference):
        guided_x = x_t.detach().clone()
        guided_x.requires_grad = True
        loss = self.mse_loss(guided_x, reference.detach()).flatten(1).sum(dim=1)
        loss.backward(torch.ones_like(loss))

        grad = guided_x.grad
        assert grad is not None
        return grad

    def denoise(self, x, t, s=None, **kwgs):
        t_batch = torch.tensor([t] * len(x), device=x.device)

        x_t_pre = self.diffusion.q_sample(x_start=x, t=t_batch)
        x_pre = self.diffusion.p_sample(
            self.model,
            x_t_pre,
            t_batch,
            clip_denoised=True,
        )["pred_xstart"]

        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x_start=x, t=t_batch, noise=noise)
        x_0_t = self.diffusion.q_sample(x_start=x_pre, t=t_batch, noise=noise)

        with TemporaryGrad():
            grad = self.guide(x_t, x_0_t)

        s = s or 0
        S = (
            s
            * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x, t_batch)
            / self.diffusion.get_sqrt_alphas_cumprod(x, t_batch)
        )
        out = self.diffusion.p_mean_variance(
            self.model,
            x_t,
            t_batch,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
        )
        var = torch.exp(out["log_variance"])
        sqrt_var = torch.exp(0.5 * out["log_variance"])
        noise = torch.randn_like(x)

        sample = (out["mean"] - S * var * grad) + sqrt_var * noise
        sample = self.diffusion.p_sample(
            self.model,
            sample,
            t_batch,
            clip_denoised=True,
        )["pred_xstart"]
        save_image((sample + 1) / 2, os.path.join("./", "guide_sample.png"))
        return sample
