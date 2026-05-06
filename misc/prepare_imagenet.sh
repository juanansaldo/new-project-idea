#!/bin/bash

# SYNOPSIS: Prepare ILSVRC2012 ImageNet into extracted train/val/test (Task 1&2 and Task 3) + devkits.
#
# USAGE EXAMPLES:
#   # Default root
#   ./prepare_imagenet.sh
#
#   # Custom root
#   ./prepare_imagenet.sh "/c/data/IMAGENET1K"

# Default root parameter
ROOT="${1:-/c/data/IMAGENET1K}"

set -e  # Exit on error

ROOT=$(realpath "$ROOT")
echo "Using root: $ROOT"

# ---- Paths to archives ----
DEVKIT_T12_TAR="$ROOT/ILSVRC2012_devkit_t12.tar.gz"
DEVKIT_T3_TAR="$ROOT/ILSVRC2012_devkit_t3.tar.gz"

TRAIN_TAR="$ROOT/ILSVRC2012_img_train.tar"
VAL_TAR="$ROOT/ILSVRC2012_img_val.tar"
TEST_TAR="$ROOT/ILSVRC2012_img_test_v10102019.tar"

TRAIN_T3_TAR="$ROOT/ILSVRC2012_img_train_t3.tar"

# ---- Target directories ----
TRAIN_DIR="$ROOT/train"
VAL_DIR="$ROOT/val"
TEST_DIR="$ROOT/test"

TRAIN_T3_DIR="$ROOT/train_t3"
DEVKIT_T12_DIR="$ROOT/devkit_t12"
DEVKIT_T3_DIR="$ROOT/devkit_t3"

echo "Preparing directories..."

for d in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR" "$TRAIN_T3_DIR" "$DEVKIT_T12_DIR" "$DEVKIT_T3_DIR"; do
    if [[ ! -d "$d" ]]; then
        echo "Creating directory: $d"
        mkdir -p "$d"
    else
        echo "Directory already exists: $d"
    fi
done

# ---- Extract devkits ----
if [[ -f "$DEVKIT_T12_TAR" ]]; then
    if [[ ! "$(ls -A "$DEVKIT_T12_DIR" 2>/dev/null)" ]]; then
        echo "Extracting devkit Task 1&2: $DEVKIT_T12_TAR -> $DEVKIT_T12_DIR"
        tar -xzf "$DEVKIT_T12_TAR" -C "$DEVKIT_T12_DIR"
    else
        echo "Devkit Task 1&2 already extracted in $DEVKIT_T12_DIR"
    fi
else
    echo "WARNING: Missing $DEVKIT_T12_TAR"
fi

if [[ -f "$DEVKIT_T3_TAR" ]]; then
    if [[ ! "$(ls -A "$DEVKIT_T3_DIR" 2>/dev/null)" ]]; then
        echo "Extracting devkit Task 3: $DEVKIT_T3_TAR -> $DEVKIT_T3_DIR"
        tar -xzf "$DEVKIT_T3_TAR" -C "$DEVKIT_T3_DIR"
    else
        echo "Devkit Task 3 already extracted in $DEVKIT_T3_DIR"
    fi
else
    echo "WARNING: Missing $DEVKIT_T3_TAR"
fi

# ---- Extract train / val / test outer tars ----
if [[ -f "$TRAIN_TAR" ]]; then
    if [[ ! "$(find "$TRAIN_DIR" -name '*.tar' -o -type d -mindepth 1 2>/dev/null)" ]]; then
        echo "Extracting train outer tar: $TRAIN_TAR -> $TRAIN_DIR"
        tar -xf "$TRAIN_TAR" -C "$TRAIN_DIR"
    else
        echo "Train outer tar seems already extracted in $TRAIN_DIR"
    fi
else
    echo "WARNING: Missing $TRAIN_TAR"
fi

if [[ -f "$VAL_TAR" ]]; then
    if [[ ! "$(find "$VAL_DIR" -name '*.JPEG' 2>/dev/null)" ]]; then
        echo "Extracting val tar: $VAL_TAR -> $VAL_DIR"
        tar -xf "$VAL_TAR" -C "$VAL_DIR"
    else
        echo "Val images already present in $VAL_DIR"
    fi
else
    echo "WARNING: Missing $VAL_TAR"
fi

if [[ -f "$TEST_TAR" ]]; then
    if [[ ! "$(find "$TEST_DIR" -name '*.JPEG' 2>/dev/null)" ]]; then
        echo "Extracting test tar: $TEST_TAR -> $TEST_DIR"
        tar -xf "$TEST_TAR" -C "$TEST_DIR"
        # Flatten: if archive had a single top-level dir (e.g. test/), move contents up
        subdirs=($(find "$TEST_DIR" -maxdepth 1 -type d ! -path "$TEST_DIR"))
        if [[ ${#subdirs[@]} -eq 1 ]]; then
            inner="${subdirs[0]}"
            echo "Flattening $(basename "$inner") -> $TEST_DIR"
            mv "$inner"/* "$TEST_DIR"/ 2>/dev/null || true
            rmdir "$inner"
        fi
    else
        echo "Test images already present in $TEST_DIR"
    fi
else
    echo "WARNING: Missing $TEST_TAR"
fi

# ---- Extract Task 3 train outer tar ----
if [[ -f "$TRAIN_T3_TAR" ]]; then
    if [[ ! "$(find "$TRAIN_T3_DIR" -name '*.tar' -o -type d -mindepth 1 2>/dev/null)" ]]; then
        echo "Extracting Task 3 train outer tar: $TRAIN_T3_TAR -> $TRAIN_T3_DIR"
        tar -xf "$TRAIN_T3_TAR" -C "$TRAIN_T3_DIR"
    else
        echo "Task 3 train outer tar seems already extracted in $TRAIN_T3_DIR"
    fi
else
    echo "WARNING: Missing $TRAIN_T3_TAR"
fi

# ---- Helper: expand class tars then delete them ----
expand_class_tars() {
    local parent_dir="$1"
    
    if [[ ! -d "$parent_dir" ]]; then
        echo "WARNING: expand_class_tars: directory does not exist: $parent_dir"
        return
    fi
    
    echo "Expanding class tars in $parent_dir ..."
    local class_tars=($(find "$parent_dir" -maxdepth 1 -name '*.tar' -type f 2>/dev/null))
    
    if [[ ${#class_tars[@]} -eq 0 ]]; then
        echo "No .tar files found in $parent_dir (maybe already expanded)."
        return
    fi
    
    echo "Expanding class tars..."
    for tar_file in "${class_tars[@]}"; do
        class_name=$(basename "$tar_file" .tar)
        target_dir="$parent_dir/$class_name"
        
        if [[ ! -d "$target_dir" ]]; then
            mkdir -p "$target_dir"
            tar -xf "$tar_file" -C "$target_dir"
        fi
        
        rm "$tar_file"
    done
    
    echo "Finished expanding class tars in $parent_dir"
}

# ---- Expand class tars for Task 1&2 and Task 3 ----
expand_class_tars "$TRAIN_DIR"
expand_class_tars "$TRAIN_T3_DIR"

echo "All done."
echo "Summary:"
echo "  Train dir:      $TRAIN_DIR"
echo "  Val dir:        $VAL_DIR"
echo "  Test dir:       $TEST_DIR"
echo "  Train T3 dir:   $TRAIN_T3_DIR"
echo "  Devkit t12 dir: $DEVKIT_T12_DIR"
echo "  Devkit t3 dir:  $DEVKIT_T3_DIR"