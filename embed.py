import numpy
import jax.numpy as np
import flax.linen as nn


class EmbeddingLearned(nn.Module):
    vocab_size: int
    embed_size: int
    max_seq_len: int

    @nn.compact
    def __call__(self, inputs):
        length = inputs.shape[-1]
        positions = np.arange(start=0, stop=length, step=1)
        word_embedding = nn.Embed(self.vocab_size, self.embed_size)(inputs)
        position_embedding = nn.Embed(self.max_seq_len, self.embed_size)(positions)
        return word_embedding + position_embedding


class EmbeddingFixed(nn.Module):
    vocab_size: int
    embed_size: int
    max_seq_len: int

    def setup(self):
        pe = numpy.zeros((self.max_seq_len, self.embed_size), dtype=numpy.float32)
        position = numpy.arange(0, self.max_seq_len)[:, numpy.newaxis]
        div_term = numpy.exp(numpy.arange(0, self.embed_size, 2) * -(numpy.log(10000.0) / self.embed_size))
        pe[:, 0::2] = numpy.sin(position * div_term)
        pe[:, 1::2] = numpy.cos(position * div_term)
        pe = pe[numpy.newaxis, :, :]  # [1, T, H]
        self.pe = np.array(pe)

    @nn.compact
    def __call__(self, x):
        word_embedding = nn.Embed(self.vocab_size, self.embed_size)(x)
        positional_embedding = self.pe[:, 0:x.shape[1], :]
        return word_embedding + positional_embedding
