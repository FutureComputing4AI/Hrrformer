import jax
import time
import optax
import jax.numpy as np
import flax.linen as nn
from functools import partial
from dataset import load_dataset
from embed import EmbeddingFixed as Embedding
from flax.training import train_state, common_utils
from HRR.with_flax import Binding, Unbinding, CosineSimilarity
from utils import save_model, save_history, grad_check, split, merge

from flax import jax_utils
from flax.training.common_utils import shard, shard_prng_key


class SelfAttention(nn.Module):
    features: int
    heads: int

    def setup(self):
        self.binding = Binding()
        self.unbinding = Unbinding()
        self.similarity = CosineSimilarity()

    @nn.compact
    def __call__(self, inputs, mask=None):
        dense = partial(nn.DenseGeneral, features=self.features, use_bias=False)

        q, k, v = (dense(name='query')(inputs), dense(name='key')(inputs), dense(name='value')(inputs))  # (B, T, H)
        q, k, v = (split(q, self.heads), split(k, self.heads), split(v, self.heads))  # (B, h, T, H')

        bind = self.binding(k, v, axis=-1)  # (B, h, T, H')
        bind = np.sum(bind, axis=-2, keepdims=True)  # (B, h, 1, H')

        vp = self.unbinding(bind, q, axis=-1)  # (B, h, T, H')
        scale = self.similarity(v, vp, axis=-1, keepdims=True)  # (B, h, T, 1)

        weight = nn.softmax(scale, axis=-2)
        weighted_value = weight * v

        weighted_value = merge(weighted_value)
        output = dense(name='output')(weighted_value)
        return output


class MLP(nn.Module):
    features: int
    dropout_rate: float
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features // 2)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.features, use_bias=False)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not self.training)
        return x


class EncoderLayer(nn.Module):
    features: int
    heads: int
    dropout_rate: float
    training: bool

    @nn.compact
    def __call__(self, x, en_mask):
        attention = SelfAttention(features=self.features, heads=self.heads)(inputs=x, mask=en_mask)
        attention = nn.Dropout(self.dropout_rate, deterministic=not self.training)(attention)
        x = nn.LayerNorm(epsilon=1e-6)(x + attention)

        ffn = MLP(self.features, self.dropout_rate, training=self.training)(x)
        return nn.LayerNorm(epsilon=1e-6)(x + ffn)


class Encoder(nn.Module):
    vocab_size: int
    max_seq_len: int
    embed_size: int
    heads: int
    n_layer: int
    dropout_rate: float
    training: bool

    @nn.compact
    def __call__(self, x, en_mask):
        x = Embedding(vocab_size=self.vocab_size,
                      embed_size=self.embed_size,
                      max_seq_len=self.max_seq_len)(x)

        x = nn.Dropout(self.dropout_rate, deterministic=not self.training)(x)

        for _ in range(self.n_layer):
            x = EncoderLayer(features=self.embed_size,
                             heads=self.heads,
                             dropout_rate=self.dropout_rate,
                             training=self.training)(x, en_mask=en_mask)

        return x


class Network(nn.Module):
    vocab_size: int
    max_seq_len: int
    embed_size: int
    heads: int
    n_layer: int
    output_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, en_in, training: bool = False):
        en_in = en_in.astype('int32')  # (B, T)

        en_out = Encoder(vocab_size=self.vocab_size,
                         max_seq_len=self.max_seq_len,
                         embed_size=self.embed_size,
                         heads=self.heads,
                         n_layer=self.n_layer,
                         dropout_rate=self.dropout_rate,
                         training=training)(en_in, en_mask=None)

        x = np.mean(en_out, axis=1)

        x = nn.Dense(self.embed_size)(x)
        x = nn.gelu(x)

        x = nn.Dense(self.output_size)(x)
        x = nn.log_softmax(x, axis=-1)
        return x


def cross_entropy_loss(true, pred):
    true = optax.smooth_labels(true, alpha=0.1)
    cross_entropy = np.sum(- true * pred, axis=-1)
    cross_entropy = np.nan_to_num(cross_entropy)
    return np.mean(cross_entropy)


def evaluate(true, pred):
    true = np.argmax(true, axis=-1)
    pred = np.argmax(pred, axis=-1)
    return np.mean(true == pred)


def initialize_model(model, max_seq_len, init_rngs):
    init_inputs = np.ones([4, max_seq_len])
    variables = model.init(init_rngs, init_inputs)['params']
    return variables


def train_step(state, batch, rngs):
    true = batch[1]
    true = common_utils.onehot(true, 10)

    def loss_fn(params):
        pred_ = state.apply_fn({'params': params}, batch[0], rngs=rngs, training=True)
        loss_ = cross_entropy_loss(true=true, pred=pred_)
        return loss_, pred_

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    grads = grad_check(grads)
    grads = jax.lax.pmean(grads, 'batch')
    state = state.apply_gradients(grads=grads)
    acc = evaluate(true=true, pred=pred)
    metrics = {'loss': loss, 'acc': acc}
    return state, metrics


def predict(state, batch, rngs):
    true = batch[1]
    true = common_utils.onehot(true, 10)
    pred = state.apply_fn({'params': state.params}, batch[0], rngs=rngs, training=False)
    acc = evaluate(true=true, pred=pred)
    metrics = {'acc': acc}
    return metrics


def train(batch_size: int, vocab_size: int, max_seq_len: int, embed_size: int, heads: int,
          n_layer: int, output_size: int, dropout_rate: float, lr: float, epochs: int):
    batch_size = batch_size * jax.local_device_count()
    name = 'image'
    print('batch size:', batch_size)

    # load dataset
    print('Loading data & building network ...')
    train_loader, _, test_loader = load_dataset(batch_size=batch_size, n_devices=1)

    # build and initialize network
    network = Network(vocab_size=vocab_size, max_seq_len=max_seq_len, embed_size=embed_size, heads=heads,
                      n_layer=n_layer, output_size=output_size, dropout_rate=dropout_rate)

    p_key_next, p_key = jax.random.split(jax.random.PRNGKey(0))
    d_key_next, d_key = jax.random.split(jax.random.PRNGKey(0))
    init_rngs = {'params': p_key, 'dropout': d_key}

    params = initialize_model(model=network, max_seq_len=max_seq_len, init_rngs=init_rngs)

    # optimizer and scheduler
    steps = (50_000 + 10_000) // batch_size
    scheduler = optax.exponential_decay(init_value=lr,
                                        transition_steps=steps,
                                        decay_rate=.95,
                                        transition_begin=1,
                                        end_value=1e-5)

    tx = optax.adam(learning_rate=scheduler)
    state = train_state.TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    # state = load_model(state, f'weights/{name}_multi_{n_layer}_{max_seq_len}.h5')
    state = jax_utils.replicate(state)
    print('Start training ...')
    history = []

    for epoch in range(1, epochs + 1):
        # train
        train_loss, train_acc = [], []
        tic = time.time()

        for data in train_loader:
            p_key_next, p_key = jax.random.split(p_key_next)
            d_key_next, d_key = jax.random.split(d_key_next)
            rngs = {'params': shard_prng_key(p_key), 'dropout': shard_prng_key(d_key)}

            x_true, y_true = data['inputs'], data['targets']
            x_true = np.reshape(x_true, (x_true.shape[0], -1))

            batch = [x_true, y_true]
            batch = shard(batch)

            state, metrics = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))(state, batch, rngs)

            if np.isnan(metrics['loss'].mean()):
                print('nan occurred!')
                continue

            train_loss.append(metrics['loss'].mean())
            train_acc.append(metrics['acc'].mean())

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc) * 100

        # test
        test_acc = []

        for data in test_loader:
            p_key_next, p_key = jax.random.split(p_key_next)
            d_key_next, d_key = jax.random.split(d_key_next)
            rngs = {'params': shard_prng_key(p_key), 'dropout': shard_prng_key(d_key)}

            x_true, y_true = data['inputs'], data['targets']
            x_true = np.reshape(x_true, (x_true.shape[0], -1))

            batch = [x_true, y_true]
            batch = shard(batch)

            metrics = jax.pmap(predict, axis_name="batch")(state, batch, rngs)

            test_acc.append(metrics['acc'].mean())

        test_acc = sum(test_acc) / len(test_acc) * 100
        toc = time.time()

        history.append(f'Epoch: [{epoch:>3d}/{epochs}], train loss: {train_loss:>8.4f}, '
                       f'train acc: {train_acc:>5.2f}%, test acc: {test_acc:>5.2f}%, '
                       f'etc: {toc - tic:>5.2f}s')
        print(history[-1])

    state = jax_utils.unreplicate(state)
    save_model(state, f'../weights/{name}_{n_layer}.h5')
    save_history(f'../weights/{name}_{n_layer}.csv', history=history, mode='w')


if __name__ == '__main__':
    import os

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    train(batch_size=2, vocab_size=256, max_seq_len=1024, embed_size=256, heads=4, n_layer=3,
          output_size=10, dropout_rate=0.1, lr=1e-3, epochs=20)
