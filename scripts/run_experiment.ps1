$experimentName = "mnist"
$configName = "config"

$env:HYDRA_FULL_ERROR = "1"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$experimentDir =(Resolve-Path ".").Path + "\experiments\$experimentName`_$timestamp"

New-Item -ItemType Directory -Force -Path $experimentDir | Out-Null

$overrides = @(
    "--config-name=$configName",
    "experiment_name=$experimentName",
    "experiment_dir=$experimentDir",
    "hydra.run.dir=$experimentDir",
    "trainer.max_epochs=5",
    "data.batch_size=256",
    "data.num_workers=0",
    "data.data_dir=C://data/MNIST"
)

python src/train.py $overrides *> "$experimentDir\run.log"