# NetFense: Adversarial Defenses against Privacy Attacks on Neural Networks for Graph Data

This is a TensorFlow implementation of the NetFense model:

NetFense can simultaneously keep graph data unnoticeability (i.e., having limited changes on the graph structure), maintain the prediction confidence of targeted label classification
(i.e., preserving data utility), and reduce the prediction confidence of private label classification (i.e., protecting the privacy of nodes).

![Goal of ARGA](https://github.com/ICHRick/NetFense/blob/main/NetFense.JPG)

We borrowed part of code from Zugner et. al.,  ̈ Adversarial attacks on neural networks for graph data." In Proceedings
of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, AAAI ’18, pages 2847–2856, 2018.

## Installation

```bash
pip install -r requirements.txt
```

## Requirements
* TensorFlow (1.0 or later)
* numpy
* scikit-learn
* scipy
* numba
## Run from

```bash
python run.py
```

## Data

In this example, we load citation network data (Citeseer). 
The original source can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and 
here (in a different format): https://github.com/danielzuegner/nettack


