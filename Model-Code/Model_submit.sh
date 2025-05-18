#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 24       # 16 cores (8 cores per GPU)
#$ -l h_rt=10:0:0    # 1 hour runtime (required to run on the short queue)
#$ -l h_vmem=11G    # 11 * 16 = 176G total RAM
#$ -l gpu=2         # request 2 GPUs
#$ -l node_type=rdg

#module load python
virtualenv /data/Environment/pytorchenv
module load python
python3 -m venv /data/Environment/pytorchenv

module load miniconda
#source /data/EECS-Theory/Clim_risk_Lab_Zahir_Rendani/Virtual-Env/pytorchenv/bin/activate
source /data/pytorchenv/bin/activate

pip install dask
pip install einops
pip install xarray
pip install timm
pip install matplotlib
pip install netcdf4
pip install --upgrade pip
pip install seaborn
pip install pip install scikit-learn
pip install torch
pip install "dask[array]"
pip install torchvision
pip install netcdf4
pip install scipy
pip install scikit-learn
pip install -U scikit-learn

python /data/UK/code/run.py
