# NIMA implementation

My implementation of [NIMA: Neural Aesthetic Assessment](https://arxiv.org/pdf/1709.05424.pdf), which is used for image quality assessment.

## Usage

```bash
# Train
$ python main.py

# Evaluate
$ python eval.py
```

The test performaces from original paper and my implementation are:

|                     | LCC_mean | SRCC_mean | EMD@1 |
|---------------------|----------|-----------|-------|
| MobileNet(In Paper) | 0.673    | 0.661     | 0.069 |
| MobileNet(My)       | 0.518    | 0.510     | 0.081 |

## Examples

![0.png](/assets/0.png)
![1.png](/assets/1.png)
![2.png](/assets/2.png)


