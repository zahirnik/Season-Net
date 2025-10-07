from torch.utils.data import TensorDataset, DataLoader

def get_dataloaders(
    input_data, target_data,
    batch_size=16,
    shuffle_train=True,
    shuffle_val=False,
    shuffle_test=False
):
    """
    Returns PyTorch DataLoaders for train, val, test splits.
    
    Args:
        input_data: torch.Tensor [season, channels, time, lat, lon]
        target_data: torch.Tensor [season, 1, time, lat, lon]
        batch_size: batch size for all loaders (default 1)
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define splits: train=0:16, val=16:20, test=20:24
    train_ds = TensorDataset(input_data[0:16], target_data[0:16])
    val_ds   = TensorDataset(input_data[16:20], target_data[16:20])
    test_ds  = TensorDataset(input_data[20:24], target_data[20:24])
    print('input_data',input_data.shape)
    print('target_data',target_data.shape)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle_val)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, val_loader, test_loader
