import os
os.environ["TORCH_USE_FLASH_ATTN"] = "0"
import torch
import numpy as np
import xarray as xr
import argparse
import random

from DataLoader import get_dataloaders
from Model import UNetConvLSTM
from engine import train_model, evaluate_model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_ensemble_input(input_data, i):
    ens = input_data[:, i, :, :, :].unsqueeze(1)   # [B, 1, T, H, W]
    features = input_data[:, 7:, :, :, :]          # [B, 18, T, H, W]
    new_input = torch.cat([ens, features], dim=1)  # [B, 19, T, H, W]
    return new_input

def make_model(device, input_channels=19, output_channels=1):
    model = UNetConvLSTM(
        input_channels=input_channels,
        output_channels=output_channels,
        features=[16, 32, 64],
        dropout=0.1,
    )
    return model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval_only"], default="train")
    parser.add_argument("--ckpt_dir", type=str, default="models_ensemble")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Region = 'NA'
    Lat_lim = range(60,124) if Region == 'AF' else range(10, 74)
    Lon_lim = range(64) if Region == 'AF' else range(230, 294)

    # === LOAD INPUT AND TARGET DATA ===
    input_data0 = xr.open_dataset('/data/.../UK_JJA_Season.nc').isel(number=list(range(7)) + list(range(25, 33)),time=range(90))
    input_data1 = xr.open_dataset('/.../UK_Prcp_JJA.nc').isel(number=range(7),time=range(90))
    input_data1['pr'][:, 0, :, :, :] = input_data1['pr'][:, 2, :, :, :]
    input_data1['pr'][:, 1, :, :, :] = input_data1['pr'][:, 2, :, :, :]
    input_data1 = input_data1.rename_vars({"pr": "t2m"}).assign_coords(number=[15, 16, 17, 18, 19, 20, 21])

    latitudes = np.linspace(90, -90, num=180)
    longitudes = np.linspace(0, 360, num=360)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    lat_da = xr.DataArray(lat_grid, dims=('latitude', 'longitude'), name='t2m')
    lon_da = xr.DataArray(lon_grid, dims=('latitude', 'longitude'), name='t2m')
    Coorrds = xr.concat([lon_da, lat_da], dim='number').assign_coords(number=[22, 23]).to_dataset(name='t2m')

    DEM = xr.open_dataset('/data/home/acw720/GMTED2010_1deg.nc').elevation
    DEM = DEM.roll(longitude=int(len(DEM.longitude)/2), roll_coords=True)
    DEM = DEM.assign_coords(longitude=[x * 0.5 for x in range(1, 720, 2)]).expand_dims(number=[24]).to_dataset(name='t2m')
    DEM = DEM.reindex(latitude=list(reversed(DEM.latitude)))

    input_data0 = xr.concat([input_data0, input_data1, Coorrds, DEM], dim='number', coords='minimal')
    target_data0 = xr.open_dataset('/data/.../ERA_JJA_Season.nc')

    input_data0 = input_data0.isel(latitude=Lat_lim, longitude=Lon_lim, time=range(90))
    target_data0 = target_data0.isel(latitude=Lat_lim, longitude=Lon_lim, time=range(90))

    # === NORMALIZE INPUT DATA ===
    List_all = []
    for i in range(25):
        ds = input_data0.isel(number=i)
        if i in [22, 23, 24]:
            maxxx, minnn = ds.t2m.max().item(), ds.t2m.min().item()
            dss = (ds - minnn) / (maxxx - minnn)
        else:
            local_min = ds.rolling(latitude=3, longitude=3, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).min(dim=['window_lat', 'window_lon', 'season', 'time'])
            local_max = ds.rolling(latitude=3, longitude=3, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).max(dim=['window_lat', 'window_lon', 'season', 'time'])
            local_min = local_min.fillna(ds.t2m)
            local_max = local_max.fillna(ds.t2m)
            local_range = local_max - local_min
            local_range = local_range.where(local_range != 0, other=1)
            dss = (ds - local_min) / local_range
        List_all.append(dss)

    # === NORMALIZE TARGET DATA ===
    train_ds = target_data0.t2m.isel(season=range(16))
    train_flat = train_ds.stack(all_time=("season", "time"))
    local_min = train_flat.rolling(latitude=3, longitude=3, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).min(dim=['window_lat', 'window_lon', 'all_time'])
    local_max = train_flat.rolling(latitude=3, longitude=3, center=True).construct({'latitude': 'window_lat', 'longitude': 'window_lon'}).max(dim=['window_lat', 'window_lon', 'all_time'])
    pixel_mean = train_flat.mean(dim='all_time')
    local_min = local_min.fillna(pixel_mean)
    local_max = local_max.fillna(pixel_mean)
    local_range = local_max - local_min
    local_range = local_range.where(local_range != 0, other=1)
    target_data = (target_data0.t2m - local_min) / local_range

    target_data = target_data.transpose("season", "time", "latitude", "longitude")
    target_data = torch.tensor(target_data.values, dtype=torch.float32).unsqueeze(1)
    input_data = xr.concat(List_all, 'number').t2m.transpose("season", "number", "time", "latitude", "longitude")
    input_data = torch.tensor(input_data.values, dtype=torch.float32)

    # === TEST SEASON SELECTION ===
    # Use last 4 seasons for test, matching your previous logic
    test_seasons_idx = np.arange(target_data0.sizes['season']-4, target_data0.sizes['season'])
    target_helper = target_data0.t2m.isel(season=test_seasons_idx)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.mode == "train":
        for ens_idx in range(7):

            print(f"==== Training for ensemble {ens_idx} ====")
            set_seed(2024 + ens_idx)
            ens_input = prepare_ensemble_input(input_data, ens_idx)
            train_loader, val_loader, test_loader = get_dataloaders(ens_input, target_data, batch_size=8)
            model = make_model(device, input_channels=19)
            model = model.float()
            trained_model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=40,
                lr=0.01,
                device=device,
                seed=2024 + ens_idx,
                model_idx=ens_idx
            )
            ckpt_path = os.path.join(args.ckpt_dir, f"model_ens{ens_idx}.pt")
            torch.save(trained_model.state_dict(), ckpt_path)
            print(f"✅ Saved model for ensemble {ens_idx} at {ckpt_path}")
    elif args.mode == "eval_only":
        all_preds_train = []
        all_preds_val = []
        all_preds_test = []
        #truth_saved = {"train": False, "val": False, "test": False}
        truth_saved = {"training": False, "validation": False, "test": False}

        unnorm = lambda arr: arr * local_range.values + local_min.values
        output_dir = os.path.abspath(f"./Exports_{Region}")
        os.makedirs(output_dir, exist_ok=True)

        for ens_idx in range(7):
            print(f"==== Evaluating model for ensemble {ens_idx} ====")
            ens_input = prepare_ensemble_input(input_data, ens_idx)
            train_loader, val_loader, test_loader = get_dataloaders(ens_input, target_data, batch_size=1)
            model = make_model(device, input_channels=19)
            ckpt_path = os.path.join(args.ckpt_dir, f"model_ens{ens_idx}.pt")
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.float()
            model.eval()

            for split_name, loader, all_preds_list in zip(
                ["Training", "Validation", "Test"],
                [train_loader, val_loader, test_loader],
                [all_preds_train, all_preds_val, all_preds_test]
            ):
                preds, targets = evaluate_model(model, loader, device=device)
                preds_unnorm = unnorm(preds)
                all_preds_list.append(torch.tensor(preds_unnorm).unsqueeze(0))

                if not truth_saved[split_name.lower()]:
                    targets_unnorm = unnorm(targets)
                    truth_xr = xr.DataArray(
                        targets_unnorm,
                        dims=("season", "time", "latitude", "longitude"),
                        coords={
                            "season": np.arange(targets_unnorm.shape[0]),
                            "time": target_helper.time.values,
                            "latitude": target_helper.latitude.values,
                            "longitude": target_helper.longitude.values
                        },
                        name="t2m"
                    )
                    truth_xr.to_netcdf(os.path.join(output_dir, f"truth_{split_name}.nc"))
                    truth_saved[split_name.lower()] = True

        # Stack and export ensemble predictions
        ensemble_dim = np.arange(7)
        for split_name, all_preds_list in zip(
            ["Training", "Validation", "Test"],
            [all_preds_train, all_preds_val, all_preds_test]
        ):
            preds_all = torch.cat(all_preds_list, dim=0)
            preds_xr = xr.DataArray(
                preds_all.numpy(),
                dims=("ensemble", "season", "time", "latitude", "longitude"),
                coords={
                    "ensemble": ensemble_dim,
                    "season": np.arange(preds_all.shape[1]),
                    "time": target_helper.time.values,
                    "latitude": target_helper.latitude.values,
                    "longitude": target_helper.longitude.values
                },
                name="t2m"
            )
            preds_xr.to_netcdf(os.path.join(output_dir, f"pred_{split_name}.nc"))

        print("✅ Saved all prediction and truth files for train, val, and test splits across 7 ensembles!")

if __name__ == "__main__":
    main()
