<h2 align="center">Hrrformer</h2>

<p align="justify">
Hrrformer is a neuro-symbolic self-attention model with linear 𝒪(n) time and space complexity. 
23× faster and consumes 24× less memory than Transformer. SOTA performance for even over sequence length n≥100,000. 
Able to learn with a single layer and converges 10× faster in LRA benchmark.
</p>

## Requirements

```Python 3.9.12```

- HRR: ```pip install --upgrade hrr```
- JAX==0.3.16 & Jaxlib==0.3.15 (
  GPU): ```pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15+cuda11.cudnn82-cp39-none-manylinux2014_x86_64.whl```
- Flax==0.6.0: ```pip install flax==0.6.0```
- Optax==0.1.2: ```pip install optax==0.1.2```
- TensorFlow (CPU): ```pip install tensorflow-cpu==2.8.0```
- TensorFlow-Datasets: ```pip install tensorflow-datasets==4.5.2```
- PyTorch (CPU): ```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu```

## Getting Started

<p align="justify">
We tested Hrrformer on EMBER and Long Range Arena (LRA) benchmarks. LRA benchmark datasets can be acquired from <a href="https://github.com/google-research/long-range-arena">LRA GitHub Page</a>. Image and Text datasets come with the TensorFlow dataset library. So, they can be used without external dependencies. All the tasks are separated into different folders. Each folder contains a data loader file along with <b>hrrformer_mgpu.py</b> file which can run the Hrrformer model in multi-GPU settings. 
</p>

```embed.py``` contains data classes for ```learned``` and ```fixed``` positional embedding. ```utils.py``` has assorted
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
