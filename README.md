# Artificial Intelligence Models for Microplastic Detection
A repository presenting optimized neural networks for micro-plastic detection tasks.

# How to Run the Experiments

## Experiments Quantized MLP with generated C code

1 - Move to the repository directoty:
```sh
$ cd /micro-plastic/detection/repo/path
```

2 - Move to the directory of quantized MLP experiments:
```sh
$ cd experiments/C/qmlp/
```

3 - Make build.sh executable:
```sh
$ chmod +x build.sh
```

4 - Build the executable:
```sh
$ ./build.sh
```

5 - Run the executable to run the experiments:
```sh
$ ./run_exp_qmlp
```
This is an output running on architecture x86_64 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz:

```sh
average: 0.00000043596000000023
max: 0.00007200000000000000
```

## Experiments Quantized MLP with generated C code

1 - Move to the repository directoty:
```sh
$ cd /micro-plastic/detection/repo/path
```

2 - Move to the directory of quantized MLP experiments:
```sh
$ cd experiments/C/qgru/
```

3 - Make build.sh executable:
```sh
$ chmod +x build.sh
```

4 - Build the executable:
```sh
$ ./build.sh
```

5 - Run the executable to run the experiments:
```sh
$ ./qgru_benchmark
```
This is an output running on architecture x86_64 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz:

```sh
average: 0.00000107914000000081
max: 0.00013600000000000000
```

## Experiments Quantized MLP with ONNX Runtime

1 - Move to the repository directoty:
```sh
$ cd /micro-plastic/detection/repo/path
```

2 - Move to the directory of quantized MLP experiments:
```sh
$ cd experiments/python/qmlp/
```

3 - Move to the directory of quantized MLP experiments:
```sh
$ python framework_template.py
```

This is an output running on architecture x86_64 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz:

```sh
average: 1.2655858993530274e-05
maximum: 0.0001380443572998047
```

## Experiments Quantized GRU with ONNX Runtime

1 - Move to the repository directoty:
```sh
$ cd /micro-plastic/detection/repo/path
```

2 - Move to the directory of quantized GRU experiments:
```sh
$ cd experiments/python/qgru/
```

3 - Move to the directory of quantized MLP experiments:
```sh
$ python ort_gru_benchmark.py
```

This is an output running on architecture x86_64 Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz:

```sh
average: 3.22261381149292e-05
maximum: 0.00012826919555664062
```

# Developers

## Branch Name

Add your branch, with the following convention:
[name]/[branch_name]

For example:
alessandro/gemm_implementation

## PR Labels

For PRs add one or more labels to classify the content:

- back-end: implementation of back-end in C.
- bug: fix a bug.
- documentation: adding documentation to the repository.
- experiments: code related to the experiments.
- front-end: implementation of models and training in Python.
- gru: code related to GRU model.
- mlp: code related to MLP model.

## PR Assignee and Reviewers

Assign the PR to yuorself, if possible.
It is possible to merge into the main branch without a review. Add someone as a review if you are unsure about the code.

## Description

Add a short description of the PR content.