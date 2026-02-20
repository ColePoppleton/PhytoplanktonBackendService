def run_pipeline():
    manager = DataManager("copernicus_data.nc")
    manager.extract_subset(spatial_ratio=0.35, temporal_ratio=0.40)
    loaders = manager.get_splits(variable="chl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwinPredictor(img_size=loaders["train"].dataset.data.shape[1])
    trainer = ModelTrainer(model, device)

    for epoch in range(1, 51):
        t_loss = trainer.train_epoch(loaders["train"])
        v_loss = trainer.validate(loaders["val"])
        trainer.log_metrics(epoch, t_loss, v_loss)

    test_loss = trainer.validate(loaders["test"])
    print(f"Final Test MSE: {test_loss:.6f}")


if __name__ == "__main__":
    run_pipeline()