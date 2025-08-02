from dataloader import AmosCTMRIDataset
from src import LatentDiffusionConditional, EMA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

train_ds = AmosCTMRIDataset()

model = LatentDiffusionConditional(train_ds, lr=1e-4, batch_size=21)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="ckpt-{global_step:06d}",
    every_n_train_steps=5000,
    save_top_k=-1,
    save_weights_only=True
)

trainer = pl.Trainer(max_steps=2e5, callbacks=[EMA(0.9999), checkpoint_callback], accelerator='gpu', devices=[0])

trainer.fit(model)
