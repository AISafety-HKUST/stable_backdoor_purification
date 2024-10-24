### Towards Stable Backdoor Purification through Feature Shift Tuning (NeurIPS 2023)
This repository contains the official implementation of [Towards Stable Backdoor Purification through Feature Shift Tuning](https://arxiv.org/abs/2310.01875) and [Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense](https://arxiv.org/abs/2410.09838).

----
<div align=center><img src=pics/framework.png  width="80%" height="60%"></div>

### Abstract

It has been widely observed that deep neural networks (DNN) are vulnerable to backdoor attacks where attackers could manipulate the model behavior maliciously by tampering with a small set of training samples. Although a line of defense methods is proposed to mitigate this threat, they either require complicated modifications to the training process or heavily rely on the specific model architecture, which makes them hard to deploy into real-world applications. Therefore, in this paper, we instead start with fine-tuning, one of the most common and easy-to-deploy backdoor defenses, through comprehensive evaluations against diverse attack scenarios. Observations made through initial experiments show that in contrast to the promising defensive results on high poisoning rates, vanilla tuning methods completely fail at low poisoning rate scenarios. Our analysis shows that with the low poisoning rate, the entanglement between backdoor and clean features undermines the effect of tuning-based defenses. Therefore, it is necessary to disentangle the backdoor and clean features in order to improve backdoor purification. To address this, we introduce Feature Shift Tuning (FST), a method for tuning-based backdoor purification. Specifically, FST encourages feature shifts by actively deviating the classifier weights from the originally compromised weights. Extensive experiments demonstrate that our FST provides consistently stable performance under different attack settings. Additionally, it is also convenient to deploy in real-world scenarios with significantly reduced computation costs.

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
#### Train backdoor models
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

#### Defense backdoor attacks
Here we demonstrate how to conduct these fine-tuning methods in our paper. For example, if you want to evaluate the feature shift tuning (FST) on backdoor models, you could use the following script:

```cmd
python fine_tune/ft.py --attack badnet --split_ratio 0.02 --pratio 0.1 \
--device cuda:0 --lr 0.01 --attack_target 0 --model resnet18 --dataset cifar10 \
--epochs 10 --ft_mode fst --alpha 0.1
```

You could further specify the tuning method by simply changing the ``` --ft_mode``` field. Currently, we support **ft** for vanilla fine-tuning; **lp** for linear-probing; **fe-tuning** for FE-tuning; **ft-init** for FT-init; **fst** for FST. 


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
```

