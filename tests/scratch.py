# %%


memory_snapshot = MemorySnapshotCallBack()
dm = BoringDataModule()
model = BoringModel()
# Initialize a trainer
trainer = pl.Trainer(
    max_epochs=1,
    callbacks=[memory_snapshot]
)
trainer.fit(model, datamodule=dm)

# %%
