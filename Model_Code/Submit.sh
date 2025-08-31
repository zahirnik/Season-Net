#!/bin/bash
#$ -pe smp 8
#$ -l h_vmem=10G
#$ -l h_rt=1:00:00
#$ -l gpu=1
##$ -l gpu_type=A100
#$ -l node_type=xdg
#$ -cwd
#$ -j y
#$ -l rocky

# ==== Load environment modules ====
module load miniforge

# ==== Set up and activate conda environment ====
ENV_NAME="pytorch2"
PYTHON_VERSION="3.10"

# Ensure conda commands available
source $(conda info --base)/etc/profile.d/conda.sh

# Only create env if it doesn't exist
if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate the environment
conda activate $ENV_NAME

# ==== Install required Python packages (one time only) ====
INSTALL_MARKER="$HOME/.installed_${ENV_NAME}_requirements"
if [ ! -f "$INSTALL_MARKER" ]; then
    echo "Installing Python packages into $ENV_NAME"
    pip install --upgrade pip
    pip install torch torchvision
    pip install xarray dask[complete] timm matplotlib netcdf4 seaborn scikit-learn einops scipy
    touch "$INSTALL_MARKER"
else
    echo "Python packages already installed for $ENV_NAME (marker present)."
fi

export TORCH_USE_FLASH_ATTN=0

cd /data/EECS-Theory/Clim_risk_Lab_Zahir_Rendani/JGR-Revised/Paper_Exports/Africa/France-SON

# ==== 1. Train all ensemble models ====
echo "==== TRAINING ENSEMBLE MODELS ===="
python run.py --mode train

# ==== 2. Run validation/test on trained models ====
echo "==== EVALUATING ENSEMBLE MODELS ===="
python run.py --mode eval_only

# ==== 3. Stack predictions from all ensemble models ====
#echo "==== STACKING ENSEMBLE OUTPUTS ===="
#python inference.py

echo "==== ALL DONE ===="
