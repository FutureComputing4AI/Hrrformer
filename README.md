<h2 align="center">Hrrformer⚡</h2>

<p align="justify">
Hrrformer is a neuro-symbolic self-attention model with linear 𝒪(T) time and space complexity. 
23× faster and consumes 24× less memory than Transformer. SOTA performance for even over sequence length T≥100,000. 
Able to learn with a single layer and converges 10× faster in LRA benchmark.
</p>

## Requirements
![requirements](https://img.shields.io/badge/Python-3.9.12-3480eb.svg?longCache=true&style=flat&logo=python)

<p align="justify">
The code is written in <a href=https://github.com/google/jax>jax</a> which is a deep learning framework develped by google. Jax leverages just-in-time (JIT) compilation and hardware acceleration to optimize the execution of numerical operations. JIT compilation is a technique that compiles code at runtime, just before it is executed which allows the compiler to optimize the code. Moreover, the numerical operations are also optimized using Accelerated Linear Algebra (XLA) compiler. Along with jax flax and optax are also used which are higher level libraries written on top of jax. 
</p>

```
pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15+cuda11.cudnn82-cp39-none-manylinux2014_x86_64.whl
pip install flax==0.6.0
pip install optax==0.1.2
```

Jax is great at optimization and making use of hardware acceleration but it does not have built in dataloader for which we have to rely on Tensorflow and PyTorch dataloaders. Install the CPU version of both of them. 

```
pip install tensorflow-cpu==2.8.0
pip install tensorflow-datasets==4.5.2
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

Finally, the library that implements the vector symbolic architecture called <a href=https://github.com/MahmudulAlam/Holographic-Reduced-Representations>Holographic Reduced Representations (HRR)</a> which is the key concept used to develop Hrrformer. 

```
pip install hrr --upgrade
```

## Dataset 
<p align="justify">
Experiments are performed on <a href=https://github.com/google-research/long-range-arena>Long Range Arena (LRA)</a> and EMBER malware classification benchmarks. To get the LRA benchmark first download the following file and extract to the working directory. Image and Text datasets come with the TensorFlow Datasets library. These datasets will be automatically downloaded while running the code. 
</p>

```
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar xvf lra_release.gz
``` 

## Getting Started
<p align="justify">
All the tasks are separated into different folders. Each folder contains a data loader file named <b>dataset.py</b> along with standalone <b>hrrformer_mgpu.py</b> file which can run the Hrrformer model in multi-GPU settings.
</p>

```embed.py``` contains data classes for ```learned``` and ```fixed``` positional embeddings. ```utils.py``` has assorted
utility files which as necessary to load/save models, write history, split/merge tensors, etc.

## Results

### EMBER

<p align="justify">
The following Figure shows the classification accuracy and the execution time for different self-attention models for incremental sequence length in the EMBER malware classification dataset. As the sequence length increases, Hrrformer outperforms the rest of the models achieving the highest 91.03% accuracy for a maximum sequence length of 16,384.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/214703845-256c9c7d-193f-47ee-9b9e-a57e1a313afa.jpg" width="600">
</p>

<p align="justify">
We use the same number of parameters as mentioned in the LRA benchmark across the tasks. Hrrformer is trained for a total of 20 epochs both in the case of single- and multi-layer which is 10x less than the previous works. The results in terms of the accuracy of the LRA benchmark are presented in the following table.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/214704340-3a54ee8d-8fee-42d5-8aa2-63d2ce6e33a7.png" width="600">
</p>

### Speed and Memory Usage
<p align="justify">
The following figure compares all the self-attention models in terms of LRA score, speed (training examples per second), and memory footprint (size of the circle). LRA score is the mean accuracy of all the tasks in the LRA benchmark. Both single- and multi-layered Hrrformer are 37x and 7.8x faster than the Luna-256 which has achieved the highest accuracy in the LRA benchmark. Hrrformer also consumes the least amount of memory, taking 76.79% and 55.24% less memory compared to Luna-256 in the case of single and multi-layered Hrrformer, respectively.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/214704525-f9bbf28c-78cb-44f7-8d5e-bd706b99520f.jpg" width="600">
</p>

### Reducing Overfitting
<p align="justify">
Hrrformer also reduces the amount of overfitting between train and test performance. The learning curves of all the task is presented in the Figure demonstrating the lower overfitting nature of the Hrrformer across the tasks. 
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/214704919-6c1c4a82-9654-44e0-b2f3-eea07d39d7f7.jpg" width="600">
</p>

### Learning 2D Structure from 1D
<p align="justify">
The ability to learn with a single layer aids in both throughput and memory use. The result is surprising, and in visualizing the weight vector W we can confirm that a single layer is sufficient to learn the structure.
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37298971/169629706-f036cf5e-73ce-47c0-8617-5ff94283edda.png" width="600">
</p>
