python pretrain.py --dataset cifar --tlabel 0 --model resnet18 --attack bench --device cuda:0 --size 32 --num_classes 10 --batch_size 128 --attack_type all2one --attack_name badnet --pratio 0.1

python btidbf.py --dataset cifar --tlabel 0 --model resnet18 --attack bench --device cuda:0 --size 32 --num_classes 10 --batch_size 128 --attack_type all2one \
--mround 20 --uround 30 --norm_bound 0.3 --attack_name badnet --pratio 0.1

python btidbfu.py --dataset cifar --tlabel 0 --model resnet18 --attack bench --device cuda:0 --size 32 --num_classes 10 --batch_size 128 --attack_type all2one \
--mround 20 --uround 30 --norm_bound 0.3 --ul_round 30 --nround 3 --attack_name badnet --pratio 0.1 --rho 0.5 --use_pam



