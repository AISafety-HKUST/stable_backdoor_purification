### Stable Backdoor Purification (NeurIPS 2023 & 2024)
This repository contains the official implementation of [Towards Stable Backdoor Purification through Feature Shift Tuning](https://arxiv.org/abs/2310.01875) and [Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense](https://arxiv.org/abs/2410.09838).

----

### Setup

Clone this repository and install all the required dependencies with the following commands.
```cmd
git clone https://github.com/AISafety-HKUST/stable_backdoor_purification.git
cd stable_backdoor_purification
conda create -n stable_backdoor_purification python=3.8
conda activate stable_backdoor_purification
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

### Pipeline
#### Train Backdoor Models
Before conducting backdoor defense, you have to train a backdoor model with the poisoned training set. Here is an example of training a [BadNet](https://arxiv.org/abs/1708.06733) model on CIFAR-10.
```cmd
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml
```
You can customize the training process by modifying the configuration files. There are two important configuration files for training a backdoor model. The configuration files in the [prototype](config/attack/prototype) directory contain some general configurations. For example, you could specify the architecture, learning rate, epoch numbers, etc by changing the corresponding field in the [file](config/attack/prototype/cifar10.yaml). For specific attacks, the configuration file lies in individual [folders](config/attack/badnet), where you could specify hyperparameters dedicated to these attacks, such as the poisoning rate and trigger type.

We also implemented the adaptive attack [Bypass](https://arxiv.org/abs/1905.13409) described in our original paper. The [Bypass](https://arxiv.org/abs/1905.13409) attack actively maximizes the indistinguishability of the hidden representations of poisoned data and clean data with adversarial regularization. We follow the methodology described in the Adversarial Embedding section and you could run the following script to attack with the [BadNet trigger](resource/badnet/trigger_image_grid.png) on CIFAR-10:
```cmd
python ./attack/badnet_bypass.py --yaml_path ../config/attack/prototype/cifar10.yaml
```
You could try the [Blend trigger](resource/blended/hello_kitty.jpeg) by simply replacing the [badnet_bypass.py](attack/badnet_bypass.py) with [blend_bypass.py](attack/blend_bypass.py).

#### Launch Feature Shift Tuning (FST)
Here we demonstrate how to conduct these fine-tuning methods to purify backdoor models. For example, if you want to evaluate the feature shift tuning (FST) on backdoor models, you could use the following script:

```cmd
python fine_tune/ft.py --attack badnet --split_ratio 0.02 --pratio 0.1 \
--device cuda:0 --lr 0.01 --attack_target 0 --model resnet18 --dataset cifar10 \
--epochs 10 --ft_mode fst --alpha 0.1 --save
```

You could further specify the tuning method by simply changing the ``` --ft_mode``` field. Currently, we support **ft** for vanilla fine-tuning; **lp** for linear-probing; **fe-tuning** for FE-tuning; **ft-init** for FT-init; **fst** for FST. 

#### Launch Retuning Attacks (RA) on Purified Models
Here we demonstrate how to conduct retuning attacks on purified models. For example, if you want to evaluate the post-robustness of FST on backdoor models, you could use the following script:

```cmd
python fine_tune/ft_poison.py --attack badnet --split_ratio 0.02 --pratio 0.1 \
--device cuda:0 --lr 0.01 --attack_target 0 --model resnet18 --dataset cifar10 \
--epochs 5 --defense_type fst --poison_num 5 --save
```
This is very similar to vanilla fine-tuning; the core difference is that the fine-tuning dataset contains backdoor examples. You could set ```--defense_type``` to specify the defense method, and set ```--poison_num``` to specify the number of backdoor examples in the fine-tuning dataset.

#### Launch Query-based Reactivation Attack (QRA)
#### Launch Path-Aware Minimization (PAM)

----
#### Our codes heavily depend on [BackdoorBench](https://github.com/SCLBD/BackdoorBench), *"BackdoorBench: A Comprehensive Benchmark of Backdoor Learning"*. It may be the best repo for backdoor research. Please consider leaving a :star: on their repository.

#### Citation

If you find our work interesting, please consider giving a star :star: and cite as:
```
@inproceedings{min2023towards,
  title={Towards Stable Backdoor Purification through Feature Shift Tuning},
  author={Min, Rui and Qin, Zeyu and Shen, Li and Cheng, Minhao},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}

@article{min2024uncovering,
  title={Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense},
  author={Min, Rui and Qin, Zeyu and Zhang, Nevin L and Shen, Li and Cheng, Minhao},
  journal={arXiv preprint arXiv:2410.09838},
  year={2024}
}
```

