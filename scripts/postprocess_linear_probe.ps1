param(
    [Parameter(Mandatory = $true)]
    [string]$ExperimentDir,

    [string]$RepoRoot = (Resolve-Path ".").Path,
    [string]$RunTag = "",
    [string]$MetricsFileName = "linear_probe_metrics.json"
)

$ErrorActionPreference = "Stop"

# Keep raw input for failure-path handling (do not Resolve-Path yet)
$ExperimentDirInput = $ExperimentDir

# Derive run tag from input path if not provided
if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = Split-Path $ExperimentDirInput -Leaf
    if ([string]::IsNullOrWhiteSpace($RunTag)) {
        $RunTag = "unknown_run"
    }
}

# Naming: avoid double-prefix like linear_probe_linear_probe_...
$reportStem = if ($RunTag -like "linear_probe_*") { $RunTag } else { "linear_probe_$RunTag" }

# Fallback manifest path (works even if experiment dir does not exist yet)
$manifestPath = Join-Path $ExperimentDirInput "manifest.json"

# Prepare default failure manifest in case anything crashes
$failManifest = [ordered]@{
    run_tag = $RunTag
    status = "failed"
    generated_at = (Get-Date).ToString("o")
    experiment_dir = $ExperimentDirInput
}

try {
    # Resolve / validate experiment dir INSIDE try so errors are caught
    if (-not (Test-Path -LiteralPath $ExperimentDirInput)) {
        throw "ExperimentDir does not exist: $ExperimentDirInput"
    }
    $ExperimentDir = (Resolve-Path -LiteralPath $ExperimentDirInput).Path

    $metricsPath = Join-Path $ExperimentDir $MetricsFileName
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        throw "Metrics file not found in experiment dir: $metricsPath"
    }

    $hydraConfigPath = Join-Path $ExperimentDir ".hydra\config.yaml"
    if (-not (Test-Path -LiteralPath $hydraConfigPath)) {
        throw "Hydra config not found: $hydraConfigPath"
    }

    $runLogPath = Join-Path $ExperimentDir "run.log"
    $manifestPath = Join-Path $ExperimentDir "manifest.json"

    # Store reports inside experiment directory
    $reportsDir = Join-Path $ExperimentDir "reports"
    $figuresDir = Join-Path $reportsDir "figures"
    $figureOut = Join-Path $figuresDir "${reportStem}_cm.png"
    $summaryPath = Join-Path $reportsDir "${reportStem}.md"

    New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null
    New-Item -ItemType Directory -Force -Path $figuresDir | Out-Null

    Write-Host "Using metrics: $metricsPath"
    Write-Host "Using hydra config: $hydraConfigPath"
    Write-Host "Figure output: $figureOut"
    Write-Host "Summary output: $summaryPath"

    # 1) Generate confusion matrix figure
    Write-Host "Generating confusion matrix..."
    python (Join-Path $RepoRoot "visualizations\reporting.py") `
        "$metricsPath" `
        --out "$figureOut" `
        --title "Linear Probe CIFAR-10 Confusion Matrix ($RunTag)"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate confusion matrix."
    }

    # 2) Parse metrics JSON
    $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    $testAcc = [double]$metrics.test_accuracy
    $report = $metrics.classification_report

    # 3) Parse .hydra/config.yaml via Python (robust and dependency-light)
    $cfgJson = python -c @"
import json, yaml, sys
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(json.dumps(cfg))
"@ "$hydraConfigPath"

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($cfgJson)) {
        throw "Failed to parse Hydra config YAML."
    }

    $cfg = $cfgJson | ConvertFrom-Json

    # Pull fields from Hydra config
    $simclrCkpt = $null
    $probeLr = $null
    $weightDecay = $null
    $batchSize = $null
    $numWorkers = $null
    $maxEpochs = $null
    $earlyStoppingPatience = "N/A"

    if ($cfg.model -and $cfg.model.pretrained_ckpt) { $simclrCkpt = [string]$cfg.model.pretrained_ckpt }
    if ($cfg.model -and $cfg.model.optimizer -and $cfg.model.optimizer.lr -ne $null) { $probeLr = [double]$cfg.model.optimizer.lr }
    if ($cfg.model -and $cfg.model.optimizer -and $cfg.model.optimizer.weight_decay -ne $null) { $weightDecay = [double]$cfg.model.optimizer.weight_decay }

    if ($cfg.datamodule -and $cfg.datamodule.batch_size -ne $null) { $batchSize = [int]$cfg.datamodule.batch_size }
    if ($cfg.datamodule -and $cfg.datamodule.num_workers -ne $null) { $numWorkers = [int]$cfg.datamodule.num_workers }
    if ($cfg.trainer -and $cfg.trainer.max_epochs -ne $null) { $maxEpochs = [int]$cfg.trainer.max_epochs }

    if ($cfg.trainer -and $cfg.trainer.callbacks) {
        foreach ($cb in $cfg.trainer.callbacks) {
            if ($cb._target_ -eq "pytorch_lightning.callbacks.EarlyStopping") {
                if ($cb.patience -ne $null) {
                    $earlyStoppingPatience = [string]$cb.patience
                } else {
                    $earlyStoppingPatience = "set (unknown)"
                }
            }
        }
    }

    # 4) Best checkpoint name from run.log (if present)
    $bestCheckpointName = "N/A"
    if (Test-Path -LiteralPath $runLogPath) {
        $line = Select-String -Path $runLogPath -Pattern "--- Testing checkpoint: best-.*\(" | Select-Object -First 1
        if ($line) {
            $txt = $line.Line
            $m = [regex]::Match($txt, "\(.*?\.ckpt\)")
            if ($m.Success) {
                $bestCheckpointName = $m.Value.Trim("()")
            } else {
                $bestCheckpointName = $txt
            }
        }
    }

    # CIFAR-10 class order
    $classNames = @(
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )

    # Build markdown table
    $classTableLines = @()
    $classTableLines += "| Class | Precision | Recall | F1-score | Support |"
    $classTableLines += "|---|---:|---:|---:|---:|"

    for ($i = 0; $i -lt $classNames.Count; $i++) {
        $key = "$i"
        if ($null -ne $report.$key) {
            $row = $report.$key
            $classTableLines += ("| {0} | {1:N4} | {2:N4} | {3:N4} | {4} |" -f `
                $classNames[$i], `
                [double]$row.precision, `
                [double]$row.recall, `
                [double]$row."f1-score", `
                [int]$row.support)
        }
    }
    $classTable = ($classTableLines -join "`n")

    $macroF1 = $null
    $weightedF1 = $null
    if ($null -ne $report."macro avg") {
        $macroF1 = [double]$report."macro avg"."f1-score"
    }
    if ($null -ne $report."weighted avg") {
        $weightedF1 = [double]$report."weighted avg"."f1-score"
    }

    # Repro commands
    $trainScriptCmd = ".\scripts\linear_probe_cifar10.ps1"
    $postScriptCmd = ".\scripts\postprocess_linear_probe.ps1 -ExperimentDir `"$ExperimentDir`""

    # 5) Write markdown summary
    $summary = @"
# Linear Probe Run Summary - $RunTag

## Artifacts
- Experiment dir: $ExperimentDir
- Metrics JSON: $metricsPath
- Hydra config: $hydraConfigPath
- Confusion matrix: $figureOut

## Key Metrics
- Test accuracy: $("{0:P2}" -f $testAcc)
- Macro F1: $(if ($null -ne $macroF1) { "{0:N4}" -f $macroF1 } else { "N/A" })
- Weighted F1: $(if ($null -ne $weightedF1) { "{0:N4}" -f $weightedF1 } else { "N/A" })

## Per-class Metrics (CIFAR-10)
$classTable

## Run Config (from .hydra/config.yaml)
- SimCLR checkpoint used: $(if ($simclrCkpt) { $simclrCkpt } else { "N/A" })
- Probe config:
  - lr: $(if ($null -ne $probeLr) { "{0:G}" -f $probeLr } else { "N/A" })
  - weight_decay: $(if ($null -ne $weightDecay) { "{0:G}" -f $weightDecay } else { "N/A" })
  - batch_size: $(if ($null -ne $batchSize) { $batchSize } else { "N/A" })
  - num_workers: $(if ($null -ne $numWorkers) { $numWorkers } else { "N/A" })
  - max_epochs: $(if ($null -ne $maxEpochs) { $maxEpochs } else { "N/A" })
  - early_stopping_patience: $earlyStoppingPatience
- Best checkpoint name: $bestCheckpointName

## Failure patterns from confusion matrix
- 
- 
- 

## Repro commands
- Train/test script: $trainScriptCmd
- Postprocess script: $postScriptCmd
"@

    $summary | Out-File -FilePath $summaryPath -Encoding UTF8

    # 6) Write success manifest.json
    $manifest = [ordered]@{
        run_tag = $RunTag
        status = "success"
        generated_at = (Get-Date).ToString("o")

        experiment_dir = $ExperimentDir
        run_log = $runLogPath
        hydra_config = $hydraConfigPath

        metrics_json = $metricsPath
        report_markdown = $summaryPath
        confusion_matrix_png = $figureOut

        model_target = if ($cfg.model -and $cfg.model._target_) { [string]$cfg.model._target_ } else { $null }
        datamodule_target = if ($cfg.datamodule -and $cfg.datamodule._target_) { [string]$cfg.datamodule._target_ } else { $null }

        simclr_checkpoint_used = $simclrCkpt
        best_checkpoint_name = $bestCheckpointName

        test_accuracy = [double]$testAcc
        macro_f1 = if ($null -ne $macroF1) { [double]$macroF1 } else { $null }
        weighted_f1 = if ($null -ne $weightedF1) { [double]$weightedF1 } else { $null }
    }

    $manifest | ConvertTo-Json -Depth 10 | Out-File -FilePath $manifestPath -Encoding UTF8

    Write-Host ""
    Write-Host "Done."
    Write-Host "Summary: $summaryPath"
    Write-Host "Figure:   $figureOut"
    Write-Host "Manifest: $manifestPath"
}
catch {
    $failManifest.error = $_.Exception.Message
    New-Item -ItemType Directory -Force -Path (Split-Path -Path $manifestPath -Parent) | Out-Null
    $failManifest | ConvertTo-Json -Depth 10 | Out-File -FilePath $manifestPath -Encoding UTF8
    Write-Error "Postprocess failed. Manifest written to: $manifestPath"
    throw
}