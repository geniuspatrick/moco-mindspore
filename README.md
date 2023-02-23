## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

This is a **unofficial** MindSpore implementation of the [MoCo paper](https://arxiv.org/abs/1911.05722):
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
It also includes the implementation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```

### Unsupervised Training

This implementation supports **multi-gpu(or npu)**, **Data Parallel Mode** training, which is faster and simpler; single-gpu(or npu) training is also supported for debugging.

To do unsupervised pre-training of a ResNet-50 model on ImageNet, run:
```shell
# 8P
python launch_moco.py --data /path/to/IN1K --output-dir ./output/moco
# 1P
python main_moco.py --data /path/to/IN1K --output-dir ./output/moco --distributed=False
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights, run:
```shell
# 8P
python launch_lincls.py --data /path/to/IN1K --output-dir ./output/lincls --pretrained ./output/moco/net.ckpt
# 1P
python main_lincls.py --data /path/to/IN1K --output-dir ./output/lincls --pretrained ./output/moco/net.ckpt --distributed=False
```
