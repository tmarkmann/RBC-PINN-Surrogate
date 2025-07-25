import hydra
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.utils.vis3D import animation_3d


@hydra.main(version_base="1.3", config_path="../configs", config_name="fno3D")
def main(config):
    dm = RBCDatamodule3D(data_dir="data/datasets/3D", **config.data)
    dm.setup(stage="test")

    idx = 0
    for inputs, ground_truth in dm.test_dataloader():
        animation_3d(
            gt=ground_truth[0].cpu().numpy(),
            pred=ground_truth[0].cpu().numpy() * 0.9,
            feature="T",
            anim_dir=config.paths.output_dir + "/animations",
            anim_name=f"test_{idx}.mp4",
        )
        idx += 1

if __name__ == "__main__":
    main()
