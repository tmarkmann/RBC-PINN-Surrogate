from pathlib import Path
import torchphysics as tp
from torchphysics.utils import grad, laplacian
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import math
import datetime


def main():
    # parameters
    Ra = 10000
    Pr = 0.7

    # coordinates
    X = tp.spaces.R1("x")
    Z = tp.spaces.R1("z")
    T = tp.spaces.R1("t")

    # fields
    U = tp.spaces.R2("u")  # velocity
    B = tp.spaces.R1("b")  # temperature
    P = tp.spaces.R1("p")  # pressure

    # domain
    X_domain = tp.domains.Interval(X, 0, 2 * math.pi)
    Z_domain = tp.domains.Interval(Z, -1, 1)
    omega = tp.domains.Parallelogram(
        X * Z, origin=[0, -1], corner_1=[0, 1], corner_2=[2 * math.pi, -1]
    )
    time = tp.domains.Interval(T, 0, 20)
    domain = time * omega

    # samplers
    def bottom_filter(z, t):
        return z[...] == -1

    def top_filter(z, t):
        return z[...] == 1

    # sampler
    inner_sampler = tp.samplers.RandomUniformSampler(
        domain, n_points=15000
    ).make_static(resample_interval=200)

    bc_bottom_sampler = tp.samplers.RandomUniformSampler(
        omega.boundary * time, filter_fn=bottom_filter, n_points=500
    ).make_static(resample_interval=200)

    bc_top_sampler = tp.samplers.RandomUniformSampler(
        omega.boundary * time, filter_fn=top_filter, n_points=500
    ).make_static(resample_interval=200)

    bc_sides_sampler = tp.samplers.RandomUniformSampler(
        Z_domain * time, n_points=500
    ).make_static(resample_interval=200)

    initial_sampler = tp.samplers.RandomUniformSampler(
        time.boundary_left * omega, n_points=5000
    ).make_static(resample_interval=200)

    # Debug plot
    # plot = tp.utils.scatter(X*Z, bc_top_sampler)
    # plt.show()
    # exit()

    # model
    model = tp.models.FCN(
        input_space=T * X * Z, output_space=U * B * P, hidden=[256] * 5
    )

    # PDE conditions
    nu = torch.sqrt(torch.tensor(Pr / Ra))
    kappa = 1.0 / torch.sqrt(torch.tensor(Pr * Ra))

    # TODO check equations

    def momentum_residual(p, b, u, x, z, t):
        return (
            grad(u, t)
            + (torch.pow(u, 2) * grad(u, x, z))
            + grad(p, x)
            - nu * laplacian(u, x, z)
            + b
        )

    def continuity_residual(u, x, z, t):
        return laplacian(u, x, z)

    def buoyancy_residual(b, u, x, z, t):
        return grad(b, t) + u * grad(b, x, z) - kappa * laplacian(b, x, z)

    momentum_cond = tp.conditions.PINNCondition(model, inner_sampler, momentum_residual)
    continuity_cond = tp.conditions.PINNCondition(
        model, inner_sampler, continuity_residual
    )
    buoyancy_cond = tp.conditions.PINNCondition(model, inner_sampler, buoyancy_residual)

    # boudary conditions
    def bc_T_bottom_residual(b):
        return b - 2

    def bc_T_top_residual(b):
        return b - 1

    def bc_noslip_residual(u):
        return u

    def bc_periodic_residual(u_left, u_right, b_left, b_right):
        return (u_left - u_right) + (b_left - b_right)

    bc_T_bottom_condition = tp.conditions.PINNCondition(
        model, bc_bottom_sampler, bc_T_bottom_residual
    )
    bc_T_top_condition = tp.conditions.PINNCondition(
        model, bc_top_sampler, bc_T_top_residual
    )
    bc_noslip_condition = tp.conditions.PINNCondition(
        model, bc_bottom_sampler + bc_top_sampler, bc_noslip_residual
    )
    bc_periodic_condition = tp.conditions.PeriodicCondition(
        module=model,
        periodic_interval=X_domain,
        non_periodic_sampler=bc_sides_sampler,
        residual_fn=bc_periodic_residual,
    )

    # initial conditions
    def initial_u_residual(u):
        return u

    def initial_b_residual(b, z):
        target = 1.5 - 0.5 * z
        return b - target

    initial_u_condition = tp.conditions.PINNCondition(
        model,
        initial_sampler,
        initial_u_residual,
        weight=50,
    )

    initial_b_condition = tp.conditions.PINNCondition(
        model,
        initial_sampler,
        initial_b_residual,
        weight=50,
    )

    # optimizer
    optim = tp.OptimizerSetting(torch.optim.Adam, lr=1e-3)
    # optim = tp.OptimizerSetting(
    #     torch.optim.LBFGS, lr=0.1, optimizer_args={"max_iter": 5}
    # )
    solver = tp.solver.Solver(
        [
            bc_T_bottom_condition,
            bc_T_top_condition,
            bc_noslip_condition,
            bc_periodic_condition,
            momentum_cond,
            continuity_cond,
            buoyancy_cond,
            initial_u_condition,
            initial_b_condition,
        ],
        optimizer_setting=optim,
    )

    # training
    logs = Path(f"logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/")

    logger = WandbLogger(
        project="rbc-pinn",
        save_dir=logs,
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        num_sanity_val_steps=0,
        benchmark=True,
        max_steps=400,
        logger=logger,
        default_root_dir=logs,
        enable_checkpointing=False,
    )
    trainer.fit(solver)

    # save the model
    # TODO

    # animation
    anim_sampler = tp.samplers.AnimationSampler(omega, time, 10, n_points=1000)

    # temperature animation
    fig, b_anim = tp.utils.animate(
        model,
        lambda b: b,
        anim_sampler,
        ani_speed=10,
        ani_type="surface_2D",
        angle=[90, -90],
    )
    b_anim.save(logs / "rbc-b.gif")
    # velocity animation
    fig, u_anim = tp.utils.animate(
        model, lambda u: u, anim_sampler, ani_speed=10, ani_type="quiver_2D"
    )
    u_anim.save(logs / "rbc-u.gif")


if __name__ == "__main__":
    main()
