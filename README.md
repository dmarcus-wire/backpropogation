# Backpropagation

- cornerstone of modern neural networks and deep learing
- 2 phases:
    - forward pass "Propogation phase": inputs are passed through network and output predictions obtained
    - backward pass "Weight update phase": compute gradient of loss function at final layer, use gradient to recursively apply chain rule to upate newtork
    
Forward pass on XOR dataset
Left XOR dataset includes class labels
Right XOR dataset includes bias column
|x0| x1| y|| x0| x1| x2|
|-|-|-|-|-|-|-|
|0|0|0||0|0|1|
|0|1|1||0|1|1|
|1|0|1||1|0|1|
|1|1|0||1|1|1|

Fully connected "FC" (Dense) layer
- inputs 1:1 to weights
- each weight is fully connected to each node in the next layer
- resulting in an output

Training two neural networks on XOR dataset and MNIST

File structure
```
% tree .
.
├── Pipefile
├── README.md
├── main.ipynb
├── nn_mnist.py
├── nn_xor.py
├── requirements.txt
└── submodules
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-37.pyc
    ├── datasets
    │   ├── __init__.py
    │   └── simpledatasetloader.py
    ├── nn
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-37.pyc
    │   │   ├── neuralnetwork.cpython-37.pyc    # most important file here
    │   │   └── perceptron.cpython-37.pyc
    │   ├── neuralnetwork.py
    │   └── perceptron.py
    └── preprocessing
        ├── __init__.py
        └── simplepreprocessor.py

```