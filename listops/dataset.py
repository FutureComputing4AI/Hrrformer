import tensorflow_datasets as tfds
from lra_benchmarks.listops.input_pipeline import get_datasets


def load_dataset(batch_size, n_devices):
    train_dataset, val_dataset, test_dataset, encoder = get_datasets(n_devices=n_devices,
                                                                     task_name='basic',
                                                                     data_dir='./../lra_release/listops-1000/',
                                                                     batch_size=batch_size,
                                                                     max_length=2000)

    train_dataset = tfds.as_numpy(train_dataset)
    valid_dataset = tfds.as_numpy(val_dataset)
    test_dataset = tfds.as_numpy(test_dataset)
    print(encoder.vocab_size)
    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    train, valid, test = load_dataset(n_devices=1, batch_size=512)

    total = 0
    for data in train:
        x = data['inputs']
        y = data['targets']
        total += x.shape[0]
        print(total)
