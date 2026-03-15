$experimentName = "simclr_imagenet"
$configName = "simclr_imagenet"

$max_epochs = "2"
$batch_size = "128"
$num_workers = "0"
$temperature = "0.5"
$dataDir = "C:/data/IMAGENET1K_tar"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$experimentDir =(Resolve-Path ".").Path + "\experiments\$experimentName`_$timestamp"

New-Item -ItemType Directory -Force -Path $experimentDir | Out-Null

$env:HYDRA_FULL_ERROR = "1"

$overrides = @(
    "--config-name=$configName",
    "experiment_name=$experimentName",
    "experiment_dir=$experimentDir",
    "hydra.run.dir=$experimentDir",
    "trainer.max_epochs=$max_epochs",
    "datamodule.batch_size=$batch_size",
    "datamodule.num_workers=$num_workers",
    "datamodule.data_dir=$dataDir",
    "model.temperature=$temperature"
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()

python src/train.py $overrides *> "$experimentDir\run.log"

$sw.Stop()
$elapsed = $sw.Elapsed
"Run completed in $($elapsed.ToString())" | Out-File -FilePath "$experimentDir\run.log" -Append