<h2 align="center">Hrrformer⚡</h2>

<p align="justify">
Hrrformer is a neuro-symbolic self-attention model with linear 𝒪(T) time and space complexity. 23× faster and consumes 24× less memory than Transformer. SOTA performance for even over sequence length T≥100,000. Able to learn with a single layer and converges 10× faster in LRA benchmark.
</p>
<p align="center">
<img src="https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/assets/37298971/ef076eaa-bace-49e6-902f-31a9518f80d7" width="1000">
</p>

## Requirements
![requirements](https://img.shields.io/badge/Python-3.9.12-3480eb.svg?longCache=true&style=flat&logo=python)

<p align="justify">
The code is written in <a href=https://github.com/google/jax>jax</a> which is a deep learning framework developed by Google. Jax leverages just-in-time (JIT) compilation and hardware acceleration to optimize the execution of numerical operations. JIT compilation is a technique that compiles code at runtime, just before it is executed which allows the compiler to optimize the code. Moreover, the numerical operations are also optimized using Accelerated Linear Algebra (XLA) compiler. Along with jax flax and optax are also used which are higher-level libraries written on top of jax. 
</p>

```properties
pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15+cuda11.cudnn82-cp39-none-manylinux2014_x86_64.whl
pip install flax==0.6.0
pip install optax==0.1.2
```

Jax is great at optimization and making use of hardware acceleration but it does not have a built-in dataloader for which we have to rely on Tensorflow and PyTorch data loaders. Install the CPU version of both of them. 

```properties
pip install tensorflow-cpu==2.8.0
pip install tensorflow-datasets==4.5.2
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

Finally, the library that implements the vector symbolic architecture called <a href=https://github.com/MahmudulAlam/Holographic-Reduced-Representations>Holographic Reduced Representations (HRR)</a> which is the key concept used to develop Hrrformer. 

```properties
pip install hrr --upgrade
```

## Dataset 
<p align="justify">
Experiments are performed on <a href=https://github.com/google-research/long-range-arena>Long Range Arena (LRA)</a> and EMBER malware classification benchmarks. To get the LRA benchmark first download the following file and extract it to the working directory. Image and Text datasets come with the TensorFlow Datasets library. These datasets will be automatically downloaded while running the code. 
</p>

```properties
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar xvf lra_release.gz
``` 

## Getting Started
<p align="justify">
All the tasks are separated into different folders. Each folder contains a data loader file named <b>dataset.py</b> along with a standalone <b>hrrformer_mgpu.py</b> file which can run the Hrrformer model in multi-GPU settings.
</p>

```embed.py``` contains data classes for ```learned``` and ```fixed``` positional embeddings. ```utils.py``` has assorted
utility files which as necessary to load/save models, write history, split/merge tensors, etc.

## Results
### LRA benchmark 
<p align="justify">
We use the same or less number of parameters as mentioned in the LRA benchmark across the tasks. Hrrformer is trained for a total of 20 epochs both in the case of single- and multi-layer which is 10x less than the previous works. The results in terms of the accuracy of the LRA benchmark are presented in the following table.
</p>

| **Model** | **ListOps** | **Text** | **Retrieval** | **Image** | **Path** | **Avg** | **Epochs** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | 54.39 | 200
Local Attention | 15.82 | 52.98 | 53.39 | 41.46 | 66.63 | 46.06 | 200
Linear Transformer | 16.13 | 65.90 | 53.09 | 42.34 | 75.30 | 50.55 | 200
Reformer | 37.27 | 56.10 | 53.40 | 38.07 | 68.50 | 50.67 | 200
Sparse Transformer | 17.07 | 63.58 | 59.59 | 44.24 | 71.71 | 51.24 | 200
Sinkhorn Transformer | 33.67 | 61.20 | 53.83 | 41.23 | 67.45 | 51.29 | 200
Linformer | 35.70 | 53.94 | 52.27 | 38.56 | 76.34 | 51.36 | 200
Performer | 18.01 | 65.40 | 53.82 | 42.77 | 77.05 | 51.41 | 200
Synthesizer | 36.99 | 61.68 | 54.67 | 41.61 | 69.45 | 52.88 | 200
Longformer | 35.63 | 62.85 | 56.89 | 42.22 | 69.71 | 53.46 | 200
BigBird | 36.05 | 64.02 | 59.29 | 40.83 | 74.87 | 55.01 | 200
F-Net | 35.33 | 65.11 | 59.61 | 38.67 | 77.78 | 54.42 | 200
Nystromformer | 37.15 | 65.52 | **79.56** | 41.58 | 70.94 | 58.95 | 200
Luna-256 | 37.98 | 65.78 | 79.56 | 47.86 | **78.55** | **61.95** | 200
H-Transformer-1D | **49.53** | **78.69** | 63.99 | 46.05 | 68.78 | 61.41 | 200
**Hrrformer Single Layer** | 38.79 | 66.50 | 75.40 | 48.47 | 70.71 | 59.97 | **20**
**Hrrformer Multi Layer** | 39.98 | 65.38 | 76.15 | **50.45** | 72.17 | 60.83 | **20**

### Speed and Memory Usage
<p align="justify">
The following figure compares all the self-attention models in terms of LRA score, speed (training examples per second), and memory footprint (size of the circle). LRA score is the mean accuracy of all the tasks in the LRA benchmark. Both single- and multi-layered Hrrformer are 28x and 10x faster than the Luna-256 which has achieved the highest accuracy in the LRA benchmark. Hrrformer also consumes the least amount of memory, taking 79.15% and 70.66% less memory compared to Luna-256 in the case of single and multi-layered Hrrformer, respectively.
</p>

<p align="center">
<img src="https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/assets/37298971/dc2a24e0-1f9b-430d-8dfd-7692723889d8" width="800">
</p>

### Learning 2D Structure from 1D
<p align="justify">
The ability to learn with a single layer aids in both throughput and memory use. The result is surprising, and in visualizing the weight vector W we can confirm that a single layer is sufficient to learn the structure.
</p>

<p align="center">
<img src="https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/assets/37298971/4872de28-8c42-4180-944a-6a2bb9a3a892" width="800">
</p>

### EMBER
<p align="justify">
The following Figure shows the classification accuracy and the execution time for different self-attention models for incremental sequence length in the EMBER malware classification dataset. As the sequence length increases, Hrrformer outperforms the rest of the models achieving the highest <b>91.03%</b> accuracy for a maximum sequence length of <b>16,384</b>.
</p>

<p align="center">
<img src="https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/assets/37298971/182d1a62-44ab-4926-83c3-859676d38d9f" width="950">
</p>

### Citations
[![Paper](https://img.shields.io/badge/ICML-2023-1495f7.svg?longCache=true&style=flat)](https://icml.cc/Conferences/2023)
[![Paper](https://img.shields.io/badge/paper-ArXiv-ff0a0a.svg?longCache=true&style=flat)](https://arxiv.org/abs/2305.19534)

To get more information about the proposed method and experiments, please go through the [paper](https://arxiv.org/abs/2305.19534). If you use this work or find this useful, cite the paper as:

```bibtex
@article{alam2023recasting,
  title={Recasting Self-Attention with Holographic Reduced Representations},
  author={Alam, Mohammad Mahmudul and Raff, Edward and Biderman, Stella and Oates, Tim and Holt, James},
  journal={arXiv preprint arXiv:2305.19534},
  year={2023}
}
```
