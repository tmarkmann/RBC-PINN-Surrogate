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
        dx = self.domain_width / width  # 2pi / 96
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

        return l1, l2, l3 + l4

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
