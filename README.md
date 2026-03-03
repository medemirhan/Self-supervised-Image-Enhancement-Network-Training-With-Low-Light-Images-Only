## Self-Supervised Low-Light Hyperspectral Image Enhancement via Fourier-Based Transformer Network
* PyTorch implementation of [SS-HSLIE](https://ieeexplore.ieee.org/document/11248865).
* Tested on pytorch==2.5.1

### Installation
* Create conda environment and install dependencies:
```
conda env create -f environment.yml
```

* Switch to the environment :
```
conda activate sshslie
```

### Train and Test Example
```
python main.py --config ./config/config.yml --model_name outdoor
```

* Configure the `config.yml` file according to your case
* The `phase` argument in the config file can be:
  * `train_and_test`
  * `train`
  * `test`

### For Only Test
* For only testing, assign the timestamp of the trained model to the `args.timestamp` in the `main.py` file. (such as `args.timestamp = '20250926_140412'`)

### Datasets
* HU-JYU datasets mentioned in the paper will be released soon.

### References
We thank to the authors of [code1](https://github.com/weichen582/RetinexNet) and [code2](https://github.com/hitzhangyu/Self-supervised-Image-Enhancement-Network-Training-With-Low-Light-Images-Only) for their implementations. Although these are implemented on TensorFlow, we utilized them as references for our codebase.