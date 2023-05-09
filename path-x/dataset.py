import tensorflow_datasets as tfds
from lra_benchmarks.image.input_pipeline import get_pathfinder_base_datasets


def load_dataset(batch_size, n_devices, resolution, split):
    train_loader, valid_loader, test_loader, _, _, inputs_shape = get_pathfinder_base_datasets(n_devices=n_devices,
                                                                                               batch_size=batch_size,
                                                                                               resolution=resolution,
                                                                                               normalize=False,
                                                                                               split=split)
    train_loader = tfds.as_numpy(train_loader)
    valid_loader = tfds.as_numpy(valid_loader)
    test_loader = tfds.as_numpy(test_loader)
    print(inputs_shape)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train, valid, test = load_dataset(batch_size=256, n_devices=1, resolution=32, split='hard')

    total = 0
    for data in train:
        x = data['inputs']
        y = data['targets']
        total += x.shape[0]
        print(total)
        break
