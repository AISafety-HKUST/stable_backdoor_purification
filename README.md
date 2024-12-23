## Backdoor Safety Tuning (NeurIPS 2023 & 2024 Spotlight)
This repository contains the official implementation of [Towards Stable Backdoor Purification through Feature Shift Tuning](https://arxiv.org/abs/2310.01875) and [Uncovering, Explaining, and Mitigating the Superficial Safety of Backdoor Defense](https://arxiv.org/abs/2410.09838).

----

### Setup

Clone this repository and install all the required dependencies with the following commands.
```cmd
git clone https://github.com/AISafety-HKUST/Backdoor_Safety_Tuning.git
cd Backdoor_Safety_Tuning
conda create -n Backdoor_Safety_Tuning python=3.8
conda activate Backdoor_Safety_Tuning
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

----

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
You could try the [Blend trigger](resource/blended/hello_kitty.jpeg) by simply replacing the [badnet_bypass.py](attack/badnet_bypass.py) with [blend_bypass.py](attack/blend_bypass.py). Note that we only implement Bypass attack on ResNet-X architecture.

#### Launch Feature Shift Tuning (FST)
Here we demonstrate how to conduct these fine-tuning methods to purify backdoor models. For example, if you want to evaluate the feature shift tuning (FST) on backdoor models, you could use the following script:

```cmd
python fine_tune/ft.py --attack badnet --split_ratio 0.02 --pratio 0.1 \
--device cuda:0 --lr 0.01 --attack_target 0 --model resnet18 --dataset cifar10 \
--epochs 10 --ft_mode fst --alpha 0.1 --save
```

You could further specify the tuning method by simply changing the ``` --ft_mode``` field. Currently, we support **ft** for vanilla fine-tuning; **lp** for linear-probing; **fe-tuning** for FE-tuning; **ft-init** for FT-init; **fst** for FST. For datasets with larger scales (e.g., CIFAR-100 and TinyImageNet), our FST may induce a significant drop in clean accuracy caused by the initialization of the linear layer. Therefore, it is recommended to use a small ```--alpha``` or simply set it to zero to maintain clean accuracy. Additionally, you could increase the number of benign examples (e.g. 5% of the training dataset) to compensate for the accuracy drop.

#### Launch Retuning Attacks (RA)
Here we demonstrate how to conduct retuning attacks (RA) on purified models. For example, if you want to evaluate the post-robustness of FST on backdoor models, you could use the following script:

```cmd
python fine_tune/ft_poison.py --attack badnet --split_ratio 0.02 --pratio 0.1 \
--device cuda:0 --lr 0.01 --attack_target 0 --model resnet18 --dataset cifar10 \
--epochs 5 --defense_type fst --poison_num 5 --save
```
This is very similar to vanilla fine-tuning; the core difference is that the fine-tuning dataset contains backdoor examples. You could set ```--defense_type``` to specify the defense method and set ```--poison_num``` to specify the number of backdoor examples in the fine-tuning dataset.

#### Launch Query-based Reactivation Attack (QRA)
Here we demonstrate how to conduct the Query-based Reactivation Attack (QRA) on purified models. For example, if you want to generate the reactivating perturbation for models purified by FST, you could use the following script:
```cmd
python fine_tune/qra.py --attack badnet --split_ratio 0.02 --pratio 0.1 \
--device cuda:0 --lr 0.01 --attack_target 0 --model resnet18 --dataset cifar10 \
--epochs 50 --defense_type fst --alpha_qra 0.2 --clean_num_qra 500 --poison_num_qra 500
```
The ```--alpha_qra``` is the parameter that controls the trade-off between reactivating and adversarial perturbation. A larger ```--alpha_qra``` indicates improved reactivating performance, but meanwhile leads to more adversarial components, which will make the reactivating perturbation attack both the backdoor and benign models simultaneously. The ```--clean_num_qra``` and ```--poison_num_qra``` indicate the number of clean and poisoned examples for optimizing the reactivating perturbation respectively.

Note that directly running this script may lead to errors; most likely, it will prompt you that the path for the clean/EP model is not found. This is because you need to train and save a clean/EP model before running this script. For example, to train a clean model, you can set the poisoning ratio to 0 with the following command:
```cmd
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml --pratio 0
```

#### Launch Path-Aware Minimization (PAM)

Here we demonstrate how to conduct PAM to purify models. First, go to the [BTIDBF folder](fine_tune/BTIDBF) and execute PAM with the following command:
```cmd
cd fine_tune/BTIDBF
sh defensh.sh
```
You can interchange BTI and PAM by setting ```--use_pam```. Our core algorithm design is depicted in this [file](fine_tune/BTIDBF/sam.py) and we provide checkpoints (four backdoor attack demos trained on CIFAR-10 and ResNet-18) robustly purified by PAM in this [link](https://drive.google.com/file/d/1qqi-ZxPFngTOwOjx8V-LTA4OP14UEt8e/view?usp=sharing). To maintain high clean accuracy and stable purification results, you may adjust the hyperparameter of BTI for better performance. Here, we list some empirical tips to improve performance. For example, if you find that PAM cannot defend against retuning attacks, try increasing the ratio of reversed examples during unlearning. Alternatively, you could increase the ```--rho``` to promote more deviation, although this may result in a drop in clean accuracy. Besides, you could slightly decrease the number of unlearning epochs specified by ```--ul_round``` (e.g., 15 and 20) and total unlearning rounds specified by ```--nround``` (e.g., for attacks like Blended and SSBA, unlearning with 1 or 2 rounds is enough) to maintain the clean performance.

----

### Citation

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
----

### Acknowledgement

#### Our codes heavily depend on [BackdoorBench](https://github.com/SCLBD/BackdoorBench) and [BTIDBF](https://github.com/xuxiong0214/BTIDBF) and please also consider leaving a :star: on their repository. Feel free to raise issues if you find bugs/reproducing problems in our code, or you can email rminaa@connect.ust.hk/zeyu.qin@connect.ust.hk.

