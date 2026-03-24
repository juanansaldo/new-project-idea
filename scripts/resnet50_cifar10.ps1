$experimentName = "resnet50_cifar10"
$configName = "resnet50_cifar10"

$max_epochs = "2"
$batch_size = "128"
$num_workers = "4"
$dataDir = "C:/data/CIFAR10"

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
    "datamodule.data_dir=$dataDir"
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()

python src/train.py $overrides *> "$experimentDir\run.log"
python src/test.py  $overrides *>> "$experimentDir\run.log"

$sw.Stop()
$elapsed = $sw.Elapsed
"Run completed in $($elapsed.ToString())" | Out-File -FilePath "$experimentDir\run.log" -Append