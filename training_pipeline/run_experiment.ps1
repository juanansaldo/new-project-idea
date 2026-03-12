$experimentName = "mnist_bs128_lr1e-4"

$overrides = @(
    "experiment_name=$experimentName",
    "data.data_dir=./data"
    "data.batch_size=128",
    "data.num_workers=0",
    "trainer.max_epochs=5",
    "trainer.devices=1",
    "model.lr=1e-4"
)

python train.py $overrides