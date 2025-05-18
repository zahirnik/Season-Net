# This script is the main run pipeline for training and evaluating a UNet2DConvLSTM model
# for seasonal bias correction using quantile mapping loss on climate data.

import os
import random
import csv
import numpy as np
import torch
import argparse
import xarray as xr
from torch.utils.data import DataLoader

# Import custom modules
import modules, unet2dconvlstm, loader, engine
from unet2dconvlstm import UNet2DConvLSTM

# Create training, validation, and test data loaders

def create_train_test_val_dataloaders(input_data, target_data, batch_size, random_state=None):
    assert 'season' in input_data.dims
    assert 'season' in target_data.dims
    seasons = input_data['season'].values

    # Split seasons into training, validation, and testing
    train_seasons = seasons[:4]
    val_seasons = seasons[4:5]
    test_seasons = seasons[5:6]

    def create_batches(input_data, target_data, selected_seasons, batch_size):
        input_batches = []
        target_batches = []
        for season in selected_seasons:
            input_season_data = input_data.sel(season=range(season * 4, (season + 1) * 4))
            target_season_data = target_data.sel(season=range(season * 4, (season + 1) * 4))
            input_tensor = torch.Tensor(input_season_data.t2m.values)
            target_tensor = torch.Tensor(target_season_data.t2m.values)
            input_batches.append(DataLoader(input_tensor, batch_size=batch_size, shuffle=False))
            target_batches.append(DataLoader(target_tensor, batch_size=batch_size, shuffle=False))
        return input_batches, target_batches

    train_input_batches, train_target_batches = create_batches(input_data, target_data, train_seasons, batch_size)
    val_input_batches, val_target_batches = create_batches(input_data, target_data, val_seasons, batch_size)
    test_input_batches, test_target_batches = create_batches(input_data, target_data, test_seasons, batch_size)

    return (train_input_batches, train_target_batches), (test_input_batches, test_target_batches), (val_input_batches, val_target_batches)

# Main function to run the pipeline
def main(args):
    # Set seeds for reproducibility
    seed = 1987
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parse command-line args
    Exp_name = args.exp_name
    batch_size = args.batch_size
    in_channels = args.in_channels
    out_channels = args.out_channels
    dropout = args.dropout
    num_filters = args.num_filters
    embd_channels = args.embd_channels
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    bottelneck_size = args.bottelneck_size
    time_window = args.time_window

    # Save random ensemble seeds to CSV
    Ens_Seed = [random.randint(1, 1000) for _ in range(7)]
    with open(f"/data/loss/Ensembles.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for number in Ens_Seed:
            writer.writerow([number])

    for ensembles in range(0, 7):
        input_data0 = xr.open_dataset("/data/out/UK_SON_Season.nc")
        target_data0 = xr.open_dataset("/data/ERA_SON_Season.nc")
        DEM_R = xr.open_dataset("/data/GMTED2010_1deg.nc")
        ds_full = xr.open_dataset("/data/Lands.nc")

        # === Process and assign DEM ===
        DEM = DEM_R.elevation.roll(longitude=180, roll_coords=True)
        DEM = DEM.assign_coords(longitude=[x * 0.5 for x in range(1, 720, 2)])
        DEM = DEM.expand_dims(number=[17]).to_dataset(name='t2m')
        DEM = DEM.reindex(latitude=list(reversed(DEM.latitude)))

        # === Assign coordinates ===
        latitudes = np.linspace(90, -90, num=180)
        longitudes = np.linspace(0, 360, num=360)
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        lat_da = xr.DataArray(lat_grid, dims=('latitude', 'longitude'), name='t2m')
        lon_da = xr.DataArray(lon_grid, dims=('latitude', 'longitude'), name='t2m')
        Coords = xr.concat([lon_da, lat_da], dim='number').assign_coords(number=[15, 16]).to_dataset(name='t2m')

        # Concatenate all into input_data0
        input_data0 = xr.concat([input_data0, Coords, DEM], dim='number')

        # Select subset
        input_data0 = input_data0.isel(latitude=range(60, 124), longitude=range(64), time=range(90))
        target_data0 = target_data0.isel(latitude=range(60, 124), longitude=range(64), time=range(90))

        # === Get land mask ===
        ds = ds_full.isel(latitude=range(60, 124), longitude=range(64)).fillna(0.001)
        weight_layer = torch.Tensor(ds.mask.values)

        # === Normalize Inputs ===
        List_all = []
        for i in range(18):
            ds = input_data0.isel(number=i)
            if i in [15, 16, 17]:
                dss = (ds - ds.t2m.min()) / (ds.t2m.max() - ds.t2m.min())
            else:
                local_min = ds.rolling(latitude=1, longitude=1, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).min(dim=['window_lat', 'window_lon', 'season', 'time'])
                local_max = ds.rolling(latitude=1, longitude=1, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).max(dim=['window_lat', 'window_lon', 'season', 'time'])
                local_range = local_max - local_min
                local_range = local_range.where(local_range != 0, other=1)
                dss = (ds - local_min) / local_range
            List_all.append(dss)

        input_data = xr.concat(List_all, 'number').isel(number=[ensembles] + list(range(7, 18)))

        # === Normalize target ===
        ds = target_data0.t2m.isel(season=range(16))
        local_min = ds.rolling(latitude=1, longitude=1, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).min(dim=['window_lat', 'window_lon', 'season', 'time'])
        local_max = ds.rolling(latitude=1, longitude=1, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).max(dim=['window_lat', 'window_lon', 'season', 'time'])
        local_range = local_max - local_min
        local_range = local_range.where(local_range != 0, other=1)
        target_data = (target_data0 - local_min) / local_range
        mini = torch.Tensor(local_min.values)
        maxi = torch.Tensor(local_max.values)
        target_season_all = torch.Tensor(target_data.sel(season=range(0, 16)).t2m.values)

        # === Dataloaders ===
        train_loader, Test_loader, Val_loader = create_train_test_val_dataloaders(input_data, target_data, batch_size=90)

        # === Model ===
        torch.manual_seed(677)
        model = UNet2DConvLSTM(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=num_filters,
            batch_size=batch_size,
            dropout=dropout,
            bottelneck_size=bottelneck_size,
            embd_channels=embd_channels,
        )

        # === Training ===
        YE = engine.BiasCorrect(model, lr=lr, wd=wd, seeded=ensembles, exp=Exp_name)
        YE.train(train_loader, Val_loader, target_season_all, weight_layer, epochs=epochs, loss_stop_tolerance=400)

        # === Prediction ===
        YE.predict(model, train_loader, mini, maxi, ensembles, Mode='train')
        YE.predict(model, Val_loader, mini, maxi, ensembles, Mode='validation')
        YE.predict(model, Test_loader, mini, maxi, ensembles, Mode='test')

# Run main if script is called directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=12)#one ensemble and the rest of fatures
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.00001)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num_filters", type=int, default=32)
    parser.add_argument("--timedim", type=int, default=24*90)
    parser.add_argument("--time_window", type=int, default=90)

    args = parser.parse_args()
    main(args)
