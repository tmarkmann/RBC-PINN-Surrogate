import hydra
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.callbacks import Example3DCallback

from rbc_pinn_surrogate.utils.vis3D import animation_3d


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_fno3D")
def main(config):
    dm = RBCDatamodule3D(data_dir="data/datasets/3D")
    dm.setup(stage="test")

    for inputs, ground_truth in dm.test_dataloader():
        animation_3d(
            gt=ground_truth.cpu().numpy(),
            pred=ground_truth.cpu().numpy(),
            feature="T",
            anim_dir=config.paths.output_dir + "/animations",
            anim_name="test.mp4",
        )
        break


if __name__ == "__main__":
    main()
