import csv
import jax
import pickle
import pandas as pd
import jax.numpy as np
from flax import serialization
from flax.core.frozen_dict import freeze, unfreeze


def split(x, heads):
    b, t, h = x.shape
    x = x.reshape(b, t, heads, h // heads)
    return x.transpose((0, 2, 1, 3))


def merge(x):
    b, heads, t, h = x.shape
    x = x.transpose((0, 2, 1, 3))
    return x.reshape(b, t, heads * h)


def look_ahead_mask(x):
    ones = np.ones(shape=(x.shape[1], x.shape[1]))
    mask = np.expand_dims(np.tril(ones), axis=0)
    return np.repeat(mask, x.shape[0], axis=0)


def one_hot(x, n_class):
    return (np.arange(n_class) == x[..., None]).astype(int)


def l2_regularization(params, alpha=1.0):
    x2 = jax.tree_map(lambda x: np.square(x).mean(), params)
    loss = np.asarray(jax.tree_leaves(x2)).sum()
    return alpha * loss


def grad_check(grads):
    grads = unfreeze(grads)
    grads = jax.tree_map(lambda x: np.nan_to_num(x), grads)
    return freeze(grads)


def index_sequence(batch_size, dataset_size):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    return list(zip(index_a, index_b))


def bias_initializer(key, shape, dtype=np.float32):
    if key is not None:
        pass
    return np.zeros(shape, dtype)


def load_model(state, name):
    with open(name, 'rb') as f:
        dict_state = pickle.loads(f.read())
    return serialization.from_state_dict(state, dict_state)


def save_model(state, name):
    dict_state = serialization.to_state_dict(state)
    with open(name, 'wb') as f:
        pickle.dump(dict_state, f)


def save_history(file, history, mode='w'):
    with open(file, mode) as f:
        writer = csv.writer(f)
        history = [line.replace(':', ',').split(',') for line in history]
        [writer.writerow(line) for line in history]


def save_history_to_csv(file, tr_acc, tr_loss, te_acc, te_loss):
    columns = ['train acc', 'train loss', 'test acc', 'test loss']
    results = pd.DataFrame(data=list(zip(tr_acc, tr_loss, te_acc, te_loss)), columns=columns)
    results.to_csv(file or None, index=False)
