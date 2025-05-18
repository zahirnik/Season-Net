# Season-Net
Season-Net is a deep learning model designed to correct biases in seasonal climate forecasts. It combines a UNet encoder-decoder with ConvLSTM to learn spatial and temporal dependencies, improving the accuracy of variables like temperature by leveraging C3S forecasts and ERA5 reanalysis as reference data.

# 🔍 Key Features

- ✅ U-Net encoder-decoder structure with skip connections
- 🔁 Spatiotemporal modeling using ConvLSTM layers
- 📉 Tailored loss functions for quantile-based bias correction
- 📊 Output diagnostics include RMSE, correlation, and probabilistic verification metrics
- 🌍 Designed for climate data grids (e.g., [Batch, Lat, Lon, Time, Features])

---

# 📁 Repository Contents

- `modules/` — Custom PyTorch modules for UNet-ConvLSTM
- `unet2dconvlstm.py` — Main model architecture
- `engine.py` — Training and validation loop
- `run.py` — Script to initialize and train Season-Net
- `figures/` — Evaluation plots and architecture image
- `README.md` — This file

---

# 🧪 Data Requirements

Season-Net was developed using:
- 📈 **Input**: C3S seasonal forecasts (e.g., 2m temperature, Z500, etc.)
- 🎯 **Target**: ERA5 reanalysis (used as ground truth)
- 🌐 Spatial resolution: 1.0° × 1.0° grid (modifiable)
- ⏱️ Temporal coverage: Monthly initializations (e.g., Feb/May/Aug/Nov), multiple lead times

---
## ⚙️ Training & Usage

Before training:
1. Prepare SEAS5 and ERA5 data in NetCDF format
2. Configure parameters in `run.py` (e.g., years, variables, batch size)
3. Launch training with:

```bash
python run.py --mode train
```

To evaluate or infer:

```bash
python run.py --mode eval
```

Model output will include corrected forecasts, evaluation scores, and figures.

---



## 📬 Contact

For questions, please contact Zahir Nikraftar at z.nikraftar@qmul.ac.uk.
