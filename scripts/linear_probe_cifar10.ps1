$experimentName = "linear_probe_cifar10"
$configName = "linear_probe_cifar10"

# Required: Lightning checkpoint from a SimCLR CIFAR-10 pretraining run
$pretrainedCkpt = "C:\Projects\AI\self-supervised-vision-lab\experiments\simclr_cifar10_20260324_122850\checkpoints\last.ckpt"

$max_epochs = "50"
$batch_size = "256"
$num_workers = "8"
$probe_lr = "1e-2"
$dataDir = "C:/data/CIFAR10"

if (-not (Test-Path -LiteralPath $pretrainedCkpt)) {
    Write-Error "pretrained_ckpt not found: $pretrainedCkpt"
    exit 1
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$experimentDir = (Resolve-Path ".").Path + "\experiments\$experimentName`_$timestamp"

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
    "model.pretrained_ckpt=$pretrainedCkpt",
    "model.optimizer.lr=$probe_lr"
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()

python src/train.py $overrides *> "$experimentDir\run.log"
python src/test.py  $overrides *>> "$experimentDir\run.log"

$sw.Stop()
$elapsed = $sw.Elapsed
"Run completed in $($elapsed.ToString())" | Out-File -FilePath "$experimentDir\run.log" -Append

Write-Host "Experiment dir: $experimentDir"