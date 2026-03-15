$dataRoot = "C:/data"
$max_epochs = "2"
$batch_size = "128"
$num_workers = "0"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$env:HYDRA_FULL_ERROR = "1"

$runs = @(
    @{ ConfigName = "config";            ExperimentName = "mnist";             DataDir = "$dataRoot/MNIST" },
    @{ ConfigName = "resnet18_cifar10";  ExperimentName = "resnet18_cifar10";  DataDir = "$dataRoot/CIFAR10" },
    @{ ConfigName = "resnet50_cifar10";  ExperimentName = "resnet50_cifar10";  DataDir = "$dataRoot/CIFAR10" },
    @{ ConfigName = "simclr_cifar10";    ExperimentName = "simclr_cifar10";    DataDir = "$dataRoot/CIFAR10" }
    # @{ ConfigName = "resnet50_imagenet"; ExperimentName = "resnet50_imagenet"; DataDir = "$dataRoot/IMAGENET1K_tar"},
    # @{ ConfigName = "simclr_imagenet";   ExperimentName = "simclr_imagenet";   DataDir = "$dataRoot/IMAGENET1k_tar"}
)

$projectRoot = (Resolve-Path ".").Path
$totalSw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($run in $runs) {
    $configName = $run.ConfigName
    $experimentName = $run.ExperimentName
    $dataDir = $run.DataDir
    $experimentDir = "$projectRoot\experiments\${experimentName}_$timestamp"

    New-Item -ItemType Directory -Force -Path $experimentDir | Out-Null

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

    Write-Host "--- Running $configName ---"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    python src/train.py $overrides *> "$experimentDir\run.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "train.py failed for $configName (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    if ($configName -notlike "simclr*") {
        python src/test.py $overrides *>> "$experimentDir\run.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "test.py failed for $configName (exit $LASTEXITCODE)"
            exit $LASTEXITCODE
        }
    } else {
        "Test skipped (SimCLR has no test dataloader)." | Out-File -FilePath "$experimentDir\run.log" -Append
    }

    $sw.Stop()
    "Run completed in $($sw.Elapsed.ToString())" | Out-File -FilePath "$experimentDir\run.log" -Append
    Write-Host "  Completed in $($sw.Elapsed.ToString())"
}

$totalSw.Stop()
Write-Host "All runs completed in $($totalSw.Elapsed.ToString())"