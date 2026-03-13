<#
.SYNOPSIS
Prepare ILSVRC2012 ImageNet into extracted train/val/test (Task 1&2 and Task 3) + devkits.

USAGE EXAMPLES:
  # Default root
  powershell -File .\prepare_imagenet.ps1

  # Custom root
  powershell -File .\prepare_imagenet.ps1 -Root "C:\data\IMAGENET1K"
#>

param(
    [string] $Root = "C:\data\IMAGENET1K"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path $Root
Write-Host "Using root: $root"

# ---- Paths to archives ----
$devkit_t12_tar = Join-Path $root "ILSVRC2012_devkit_t12.tar.gz"
$devkit_t3_tar  = Join-Path $root "ILSVRC2012_devkit_t3.tar.gz"

$train_tar      = Join-Path $root "ILSVRC2012_img_train.tar"
$val_tar        = Join-Path $root "ILSVRC2012_img_val.tar"
$test_tar       = Join-Path $root "ILSVRC2012_img_test_v10102019.tar"

$train_t3_tar   = Join-Path $root "ILSVRC2012_img_train_t3.tar"

# ---- Target directories ----
$train_dir      = Join-Path $root "train"
$val_dir        = Join-Path $root "val"
$test_dir       = Join-Path $root "test"

$train_t3_dir   = Join-Path $root "train_t3"
$devkit_t12_dir = Join-Path $root "devkit_t12"
$devkit_t3_dir  = Join-Path $root "devkit_t3"

Write-Host "Preparing directories..."

foreach ($d in @($train_dir, $val_dir, $test_dir, $train_t3_dir, $devkit_t12_dir, $devkit_t3_dir)) {
    if (-not (Test-Path $d)) {
        Write-Host "Creating directory: $d"
        New-Item -ItemType Directory -Path $d | Out-Null
    } else {
        Write-Host "Directory already exists: $d"
    }
}

# ---- Extract devkits ----
if (Test-Path $devkit_t12_tar) {
    if (-not (Get-ChildItem $devkit_t12_dir -ErrorAction SilentlyContinue)) {
        Write-Host "Extracting devkit Task 1&2: $devkit_t12_tar -> $devkit_t12_dir"
        tar -xzf $devkit_t12_tar -C $devkit_t12_dir
    } else {
        Write-Host "Devkit Task 1&2 already extracted in $devkit_t12_dir"
    }
} else {
    Write-Warning "Missing $devkit_t12_tar"
}

if (Test-Path $devkit_t3_tar) {
    if (-not (Get-ChildItem $devkit_t3_dir -ErrorAction SilentlyContinue)) {
        Write-Host "Extracting devkit Task 3: $devkit_t3_tar -> $devkit_t3_dir"
        tar -xzf $devkit_t3_tar -C $devkit_t3_dir
    } else {
        Write-Host "Devkit Task 3 already extracted in $devkit_t3_dir"
    }
} else {
    Write-Warning "Missing $devkit_t3_tar"
}

# ---- Extract train / val / test outer tars ----
if (Test-Path $train_tar) {
    if (-not (Get-ChildItem $train_dir -Filter '*.tar' -ErrorAction SilentlyContinue) -and
        -not (Get-ChildItem $train_dir -Directory -ErrorAction SilentlyContinue)) {
        Write-Host "Extracting train outer tar: $train_tar -> $train_dir"
        tar -xf $train_tar -C $train_dir
    } else {
        Write-Host "Train outer tar seems already extracted in $train_dir"
    }
} else {
    Write-Warning "Missing $train_tar"
}

if (Test-Path $val_tar) {
    if (-not (Get-ChildItem $val_dir -Filter '*.JPEG' -ErrorAction SilentlyContinue)) {
        Write-Host "Extracting val tar: $val_tar -> $val_dir"
        tar -xf $val_tar -C $val_dir
    } else {
        Write-Host "Val images already present in $val_dir"
    }
} else {
    Write-Warning "Missing $val_tar"
}

if (Test-Path $test_tar) {
    if (-not (Get-ChildItem $test_dir -Filter '*.JPEG' -ErrorAction SilentlyContinue)) {
        Write-Host "Extracting test tar: $test_tar -> $test_dir"
        tar -xf $test_tar -C $test_dir
    } else {
        Write-Host "Test images already present in $test_dir"
    }
} else {
    Write-Warning "Missing $test_tar"
}

# ---- Extract Task 3 train outer tar ----
if (Test-Path $train_t3_tar) {
    if (-not (Get-ChildItem $train_t3_dir -Filter '*.tar' -ErrorAction SilentlyContinue) -and
        -not (Get-ChildItem $train_t3_dir -Directory -ErrorAction SilentlyContinue)) {
        Write-Host "Extracting Task 3 train outer tar: $train_t3_tar -> $train_t3_dir"
        tar -xf $train_t3_tar -C $train_t3_dir
    } else {
        Write-Host "Task 3 train outer tar seems already extracted in $train_t3_dir"
    }
} else {
    Write-Warning "Missing $train_t3_tar"
}

# ---- Helper: expand class tars then delete them ----
function Expand-ClassTars {
    param(
        [Parameter(Mandatory = $true)]
        [string] $ParentDir
    )

    if (-not (Test-Path $ParentDir)) {
        Write-Warning "Expand-ClassTars: directory does not exist: $ParentDir"
        return
    }

    Write-Host "Expanding class tars in $ParentDir ..."
    $classTars = Get-ChildItem $ParentDir -Filter '*.tar' -File -ErrorAction SilentlyContinue
    if (-not $classTars) {
        Write-Host "No .tar files found in $ParentDir (maybe already expanded)."
        return
    }

    Write-Host "Expanding class tars..."
    foreach ($tarFile in $classTars) {

        $className = [System.IO.Path]::GetFileNameWithoutExtension($tarFile.Name)
        $targetDir = Join-Path $ParentDir $className

        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir | Out-Null
            tar -xf $tarFile.FullName -C $targetDir
        }

        Remove-Item $tarFile.FullName
    }

    Write-Host "Finished expanding class tars in $ParentDir"
}

# ---- Expand class tars for Task 1&2 and Task 3 ----
Expand-ClassTars -ParentDir $train_dir
Expand-ClassTars -ParentDir $train_t3_dir

Write-Host "All done."
Write-Host "Summary:"
Write-Host "  Train dir:      $train_dir"
Write-Host "  Val dir:        $val_dir"
Write-Host "  Test dir:       $test_dir"
Write-Host "  Train T3 dir:   $train_t3_dir"
Write-Host "  Devkit t12 dir: $devkit_t12_dir"
Write-Host "  Devkit t3 dir:  $devkit_t3_dir"