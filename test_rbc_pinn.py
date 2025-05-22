import torchphysics as tp


# TODO load model


# We can also animate the solution over time
# anim_sampler = tp.samplers.AnimationSampler(omega, I_t, 200, n_points=1000)
# fig, anim = tp.utils.animate(model, lambda u: u, anim_sampler, ani_speed=10, angle=[30, 220])
# anim.save('heat-eq.gif')