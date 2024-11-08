import torch
from torchvision import transforms
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from models import densenet, resnet, vgg, densenet_mod, resnet_mod, resnet20
from models import vit
from models.ia.models import Generator
import cifar
import cifar100
import tiny

class Box():
    def __init__(self, opt) -> None:
        self.opt = opt
        self.dataset = opt.dataset
        self.tlabel = opt.tlabel
        self.model = opt.model
        self.attack = opt.attack
        self.normalizer = self.get_normalizer()
        self.denormalizer = self.get_denormalizer()
        self.size = opt.size
        self.device = opt.device
        self.num_classes = opt.num_classes
        self.attack_type = opt.attack_type
        self.root = opt.root
        if self.attack_type == "all2all":
            self.res_path = self.dataset + "-" + self.attack + "-" + self.model + "-targetall"
        elif self.attack_type == "all2one":
            self.res_path = self.dataset + "-" + self.attack + "-" + self.model + "-target" + str(self.tlabel)
    
    def get_save_path(self):
        save_path = os.path.join(self.root, "results/"+self.res_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return save_path


    def get_normalizer(self):
        if self.dataset == "cifar":
            # return transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            return transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            
        elif self.dataset == "gtsrb":
            return transforms.Normalize([0, 0, 0], [1, 1, 1])
        elif self.dataset == "imagenet" or self.dataset == "cifar100" or self.dataset == "tiny":
            return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            raise Exception("Invalid dataset")
        
    def get_denormalizer(self):
        if self.dataset == "cifar":
            return transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        elif self.dataset == "gtsrb":
            return transforms.Normalize([0, 0, 0], [1, 1, 1])
        elif self.dataset == "imagenet" or self.dataset == "cifar100" or self.dataset == "tiny":
            return transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
        else:
            raise Exception("Invalid dataset")
        
    def get_transform(self, train):
        if train == "clean" or train == "poison":
            if self.dataset == "cifar":
                return transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
            
            elif self.dataset == "imagenet" or self.dataset == "cifar100" or self.dataset == "tiny":
                return transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.RandomCrop(size=224, padding=4),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
            elif self.dataset == "gtsrb":
                return transforms.Compose([transforms.Resize((40, 40)),
                                           transforms.RandomCrop(size=32, padding=4),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0, 0, 0], [1, 1, 1])])
            else:
                raise Exception("Invalid dataset")
        
        elif train == "test":
            if self.dataset == "cifar":
                return transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
            
            elif self.dataset == "imagenet" or self.dataset == "cifar100" or self.dataset == "tiny":
                return transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.CenterCrop(size=224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            elif self.dataset == "gtsrb":
                return transforms.Compose([transforms.Resize((40, 40)),
                                           transforms.CenterCrop(size=32),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0, 0, 0], [1, 1, 1])])
            else:
                raise Exception("Invalid dataset")
        
        else:
            raise Exception("Invalid train")

    def poisoned(self, img_tensor, param1=None, param2=None):
        if self.attack == "badnets":
            mask = param1
            ptn = param2
            img_tensor = self.denormalizer(img_tensor)
            bd_inputs = (1-mask) * img_tensor + mask*ptn
            return self.normalizer(bd_inputs)
        elif self.attack == "blend":
            alpha = param1
            trigger = param2
            bd_inputs = (1-alpha) * img_tensor + alpha*self.normalizer(trigger)
            return bd_inputs
        elif self.attack == "wanet":
            noise_grid = param1
            identity_grid = param2
            grid_temps = (identity_grid + 0.5 * noise_grid / self.size) * 1
            grid_temps = torch.clamp(grid_temps, -1, 1)
            num_bd = img_tensor.shape[0]
            bd_inputs = F.grid_sample(img_tensor[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
            return bd_inputs
        elif self.attack == "ia":
            netG = param1
            netM = param2
            patterns = netG(img_tensor)
            patterns = netG.normalize_pattern(patterns)
            masks_output = netM.threshold(netM(img_tensor))
            bd_inputs = img_tensor + (patterns - img_tensor) * masks_output
            return bd_inputs
        elif self.attack == "lc":
            mask = param1
            ptn = param2
            img_tensor = self.denormalizer(img_tensor)
            bd_inputs = (1-mask) * img_tensor + mask*ptn
            return self.normalizer(bd_inputs)
        elif self.attack == "bppattack":
            inputs_bd = self.back_to_np_4d(img_tensor, self.opt)
            squeeze_num = 8
            inputs_bd = torch.round(inputs_bd/255.0*(squeeze_num-1))/(squeeze_num-1)*255
            inputs_bd = self.np_4d_to_tensor(inputs_bd,self.opt)
            return inputs_bd
        elif self.attack == "bench":
            
            return img_tensor

        else:
            raise Exception("Invalid attack")

    def get_dataloader(self, train, batch_size, shuffle):
        tf = self.get_transform(train)
        if self.dataset == "cifar":
            if train == "clean":
                ds = cifar.CIFAR(path=os.path.join(self.root, "datasets/cifar10"), train=True, train_type=0, tf=tf)
            elif train == "poison":
                ds = cifar.CIFAR(path=os.path.join(self.root, "datasets/cifar10"), train=True, train_type=1, tf=tf)
            else:
                ds = cifar.CIFAR(path=os.path.join(self.root, "datasets/cifar10"), train=False, tf=tf)

        if self.dataset == "cifar100":
            if train == "clean":
                ds = cifar100.CIFAR(path=os.path.join(self.root, "../BackdoorBench-main/data/cifar100"), train=True, train_type=0, tf=tf)
            elif train == "poison":
                ds = cifar100.CIFAR(path=os.path.join(self.root, "../BackdoorBench-main/data/cifar100"), train=True, train_type=1, tf=tf)
            else:
                ds = cifar100.CIFAR(path=os.path.join(self.root, "../BackdoorBench-main/data/cifar100"), train=False, tf=tf)

        if self.dataset == "tiny":
            if train == "clean":
                ds = tiny.TinyImageNet(path=os.path.join(self.root, "../BackdoorBench-main/data/tiny"), train=True, train_type=0, tf=tf)
           
            else:
                ds = tiny.TinyImageNet(path=os.path.join(self.root, "../BackdoorBench-main/data/tiny"), train=False, tf=tf)
        
        # elif self.dataset == "imagenet":
        #     ds = imagenet.ImageNet(path=os.path.join(self.root, "datasets"), train=train, tf=tf)
        
        # elif self.dataset == "gtsrb":
        #     if train == "clean":
        #         ds = gtsrb.GTSRB(path=os.path.join(self.root, "datasets/gtsrb"), train=True, train_type=0, tf=tf)
        #     elif train == "poison":
        #         ds = gtsrb.GTSRB(path=os.path.join(self.root, "datasets/gtsrb"), train=True, train_type=1, tf=tf)
        #     else:
        #         ds = gtsrb.GTSRB(path=os.path.join(self.root, "datasets/gtsrb"), train=False, tf=tf)

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=6)
        return dl

    def get_state_dict(self, path=None):
        if self.attack_type == "all2one":
            filename = self.dataset + "-" + self.attack + "-" + self.model + "-target" + str(self.tlabel) + ".pt.tar"
        elif self.attack_type == "all2all":
            filename = self.dataset + "-" + self.attack + "-" + self.model + "-targetall.pt.tar"
        else:
            raise Exception("Invalid Attack Type")
        if path is None:
            state_dict = torch.load(os.path.join(self.root, "checkpoints/"+filename), map_location=torch.device('cpu'))
        elif self.attack == 'bench' and path is not None:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
    
        if self.attack == "badnets":
            param1 = state_dict["mask"]
            param2 = state_dict["ptn"]
        elif self.attack == "lc":
            param1 = state_dict["mask"]
            param2 = state_dict["ptn"]
        elif self.attack == "blend":
            param1 = state_dict["alpha"]
            param2 = state_dict["trigger"]
        elif self.attack == "wanet":
            param1 = state_dict["noise_grid"]
            param2 = state_dict["identity_grid"]
        elif self.attack == "ia":
            param1 = Generator(dataset=self.dataset)
            param2 = Generator(dataset=self.dataset, out_channels=1)
            param1.load_state_dict(state_dict["netG"])
            param2.load_state_dict(state_dict["netM"])
            param1.eval()
            param2.eval()
        elif self.attack == "bppattack":
            param1 = None
            param2 = None
        elif self.attack == "bench":
            param1 = None
            param2 = None
        else:
            raise Exception("Invalid attack")

        classifier = self.get_model()
        try:
            classifier.load_state_dict(state_dict["netC"])
        except:
            if 'model' in state_dict.keys():
                classifier.load_state_dict(state_dict["model"])
            else:
                classifier.load_state_dict(state_dict)
        
        classifier = classifier.to(self.device)
        classifier.eval()

        try:
            param1 = param1.to(self.device)
        except:
            pass
        try:
            param2 = param2.to(self.device)
        except:
            pass

        return param1, param2, classifier
    
    def get_model(self):

        if self.model == "densenet":
            return densenet.DenseNet121(num_classes=self.num_classes)
        
        elif self.model == "resnet18" and self.opt.pre==False:
            return resnet.ResNet18(num_classes=self.num_classes)
        elif self.model == "resnet20" and self.opt.pre==False:
            return resnet20.resnet20(num_classes=self.num_classes)

        elif self.model == "resnet18" and self.opt.pre:
            if self.opt.dataset == 'cifar100':
                return resnet_mod.ResNet18(num_classes=100)
            elif self.opt.dataset == 'tiny':
                return resnet_mod.ResNet18(num_classes=200)

        elif self.model == "resnet50":
            return resnet.ResNet50(num_classes=self.num_classes)
            
        elif self.model == "vgg16":
            return vgg.VGG("VGG16", num_classes=self.num_classes)
        elif self.model == 'densenet161':
            return densenet_mod.DenseNet161(num_classes=self.num_classes)
            
        elif self.model == "vit":
            return vit.ViT(image_size = self.size,
                           patch_size = 4,
                           num_classes = self.num_classes,
                           dim = int(512),
                           depth = 6,
                           heads = 8,
                           mlp_dim = 512,
                           dropout = 0.1,
                           emb_dropout = 0.1)

    # BppAttak tools
    def back_to_np_4d(self, inputs, opt):
        if opt.dataset == "cifar":
            expected_values = [0.4914, 0.4822, 0.4465]
            variance = [0.247, 0.243, 0.261]
        elif opt.dataset == "mnist":
            expected_values = [0.5]
            variance = [0.5]
        elif opt.dataset == "imagenet":
            expected_values = [0.485, 0.456, 0.406]
            variance = [0.229, 0.224, 0.225]
        elif opt.dataset in ["gtsrb","celeba"]:
            expected_values = [0,0,0]
            variance = [1,1,1]
        inputs_clone = inputs.clone()
        if opt.dataset == "mnist":
            inputs_clone[:,:,:,:] = inputs_clone[:,:,:,:] * variance[0] + expected_values[0]

        else:
            for channel in range(3):
                inputs_clone[:,channel,:,:] = inputs_clone[:,channel,:,:] * variance[channel] + expected_values[channel]

        return inputs_clone*255
    
    def np_4d_to_tensor(self, inputs, opt):
        if opt.dataset == "cifar":
            expected_values = [0.4914, 0.4822, 0.4465]
            variance = [0.247, 0.243, 0.261]
        elif opt.dataset == "mnist":
            expected_values = [0.5]
            variance = [0.5]
        elif opt.dataset == "imagenet":
            expected_values = [0.485, 0.456, 0.406]
            variance = [0.229, 0.224, 0.225]
        elif opt.dataset in ["gtsrb","celeba"]:
            expected_values = [0,0,0]
            variance = [1,1,1]
        inputs_clone = inputs.clone().div(255.0)

        if opt.dataset == "mnist":
            inputs_clone[:,:,:,:] = (inputs_clone[:,:,:,:] - expected_values[0]).div(variance[0])
        else:
            for channel in range(3):
                inputs_clone[:,channel,:,:] = (inputs_clone[:,channel,:,:] - expected_values[channel]).div(variance[channel])

        return inputs_clone
    
