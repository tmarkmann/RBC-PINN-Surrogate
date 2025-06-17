from rbc_pinn_surrogate.data import RBCDataset, RBCDatamodule
from pathlib import Path

path = Path("data/2D/val/ra10000.h5")
print(path.exists())

dm = RBCDatamodule(data_dir="data/2D")

dm.setup("fit")
dl = dm.train_dataloader()