import torch
from torch import nn
from torch.nn.functional import mse_loss


class RBCEquationLoss(nn.Module):
    """expects data in the form [batch, channel, time, height, width]"""

    def __init__(self, domain_width, domain_height, time, kappa, nu):
        super().__init__()
        self.domain_width = domain_width
        self.domain_height = domain_height
        self.time = time
        self.kappa = kappa  # thermal conductivity
        self.nu = nu  # kinematic viscosity

    def forward(self, pred):
        # discretization steps
        _, _, time_steps, height, width = pred.shape
        dt = self.time / time_steps
        dx = self.domain_width / width  # 6pi / 96
        dz = self.domain_height / height  # 2 / 64

        # fields
        T = pred[:, 0].squeeze(1)
        u = pred[:, 1].squeeze(1)
        w = pred[:, 2].squeeze(1)

        # hydrostatic + non-hydrostatic pressure
        p = pred[:, 3].squeeze(1) + pred[:, 4].squeeze(1)

        # equations
        l1 = self.loss_incompressible(u, w, dx, dz)
        l2 = self.loss_temperature(T, dt, u, dx, w, dz)
        l3, l4 = self.loss_momentum(u, w, p, T, dt, dx, dz)

        return l1 + l2 + l3 + l4

    def loss_incompressible(self, u, w, dx, dz):
        """
        u,x have shape [batch, time, height, width]
        """
        # TODO look into gradient dir
        # compute div = du/dx + dw/dz = 0
        dudx = torch.gradient(u, dim=3, spacing=dx)[0]
        dwdz = torch.gradient(w, dim=2, spacing=dz)[0]
        div = dudx + dwdz

        return mse_loss(div, torch.zeros_like(div))

    def loss_temperature(self, T, dt, u, dx, w, dz):
        # equation dT/dt + u * dT/dx + w * dT/dz = k * dT/dxx + dT/dzz

        # compute partial derivatives of T
        dTdt, dTdz, dTdx = torch.gradient(T, dim=[1, 2, 3], spacing=[dt, dz, dx])
        left = dTdt + u * dTdx + w * dTdz

        # compute second derivatives of T
        dTdxx = torch.gradient(dTdx, dim=3, spacing=dx)[0]
        dTdzz = torch.gradient(dTdz, dim=2, spacing=dz)[0]
        right = self.kappa * (dTdxx + dTdzz)

        return mse_loss(left, right)

    def loss_momentum(self, u, w, p, T, dt, dx, dz):
        # compute partial derivatives of u
        dudt, dudz, dudx = torch.gradient(u, dim=[1, 2, 3], spacing=[dt, dz, dx])

        # compute partial derivatives of w
        dwdt, dwdz, dwdx = torch.gradient(w, dim=[1, 2, 3], spacing=[dt, dz, dx])

        # compute pressure gradient
        dpdz, dpdx = torch.gradient(p, dim=[2, 3], spacing=[dz, dx])

        # compute second derivatives of u
        dudxx = torch.gradient(dudx, dim=3, spacing=dx)[0]
        dudzz = torch.gradient(dudz, dim=2, spacing=dz)[0]

        # compute second derivatives of w
        dwdxx = torch.gradient(dwdx, dim=3, spacing=dx)[0]
        dwdzz = torch.gradient(dwdz, dim=2, spacing=dz)[0]

        # x component equation: du/dt + u * du/dx + w * du/dz = -dp/dx + nu * (d2u/dxx + d2u/dzz)
        x_left = dudt + u * dudx + w * dudz
        x_right = -dpdx + self.nu * (dudxx + dudzz)

        # z component equation: dw/dt + u * dw/dx + w * dw/dz = -dp/dz + nu * (d2w/dxx + d2w/dzz) + T
        z_left = dwdt + u * dwdx + w * dwdz
        z_right = -dpdz + self.nu * (dwdxx + dwdzz) + T

        return mse_loss(x_left, x_right), mse_loss(z_left, z_right)

    def plot_div(self, div):
        import matplotlib.pyplot as plt

        plt.imshow(
            div.cpu().detach().numpy(),
            cmap="hot",
            interpolation="nearest",
            origin="lower",
        )
        plt.colorbar()
        plt.title("Divergence of velocity field")
        plt.show()


def _k_grids(shape, spacings, device, dtype):
    """
    Build wave-number grids kx, ky, (kz) using FFT conventions.
    shape: tuple of spatial sizes, e.g. (H, W) or (D, H, W) in the same order as FFT dims
    spacings: tuple of dx for each dim in the same order as 'shape'
    """
    ks = []
    for n, d in zip(shape, spacings):
        # 2π * frequency; tfft.fftfreq uses cycles/unit, multiply by 2π for angular freq
        ks.append(
            2.0 * torch.pi * torch.fft.fftfreq(n, d=d, device=device, dtype=dtype)
        )
    # Meshgrid in 'ij' indexing to match tensor layout
    return torch.meshgrid(*ks, indexing="ij")  # returns k0, k1, (k2)


def divergence_free_loss(u, spacings, fft_dims=None, eps=1e-12):
    """
    u: velocity tensor of shape
       - 2D: (B, 2, H, W)      or
       - 3D: (B, 3, D, H, W)
    spacings: tuple of grid spacings along the FFT'd spatial dims, e.g. (dz, dy, dx) matching the order of fft_dims
    fft_dims: which tensor dims to FFT over (default = last 2 or last 3, inferred from u.shape[1:])
              e.g. for (B, C, H, W) -> fft_dims=(-2, -1); for (B, C, D, H, W) -> (-3, -2, -1)
    Returns: mean squared Fourier divergence (real scalar)
    """
    assert u.dim() in (4, 5), "u must be (B, C, H, W) or (B, C, D, H, W)"
    B, C = u.shape[:2]
    is3d = u.dim() == 5
    assert C in (2, 3), "velocity channels must be 2 (2D) or 3 (3D)"

    if fft_dims is None:
        fft_dims = tuple(range(-2, 0)) if not is3d else tuple(range(-3, 0))

    # Move the FFT dims to the end (contiguity helps but not required)
    # We'll just call fft over fft_dims directly.
    device, real_dtype = u.device, u.dtype

    # FFT of each velocity component
    Uc = torch.fft.fftn(u, dim=fft_dims)  # complex

    # Build k-grids in the order of fft_dims
    spatial_shape = tuple(u.shape[d] for d in fft_dims)
    k_grids = _k_grids(spatial_shape, spacings, device, real_dtype)
    # Broadcast k-grids to full tensor shape for dot product with U
    # We need them with leading singleton dims for (B, C, ...spatial...)
    # k_grids[i] has shape spatial_shape; unsqueeze to (1, 1, ...spatial...)
    k_expanded = [kg.reshape((1, 1) + kg.shape) for kg in k_grids]

    # Compute i * k · U_hat  (Fourier divergence)
    # Uc has shape (B, C, ...spatial...), channels ordered (u_x, u_y, u_z)
    if C == 2:
        kdotU = k_expanded[0] * Uc[:, 0] + k_expanded[1] * Uc[:, 1]
    else:  # C == 3
        kdotU = (
            k_expanded[0] * Uc[:, 0]
            + k_expanded[1] * Uc[:, 1]
            + k_expanded[2] * Uc[:, 2]
        )
    div_hat = 1j * kdotU  # complex

    # Optionally ignore the k=0 mode to avoid trivial constant-mode contribution
    k2 = sum(ke**2 for ke in k_expanded)  # shape (1,1,...)
    mask = k2 > eps

    # Mean squared magnitude of divergence (only nonzero modes)
    num = (div_hat.abs() ** 2 * mask).sum(dtype=real_dtype)
    den = mask.sum(dtype=real_dtype).clamp_min(1.0)
    return num / den
