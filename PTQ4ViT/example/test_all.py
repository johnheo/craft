from test_vit import *

from itertools import product

from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
import time
import copy

# suppress user warning
import warnings
warnings.filterwarnings("ignore")
PATH = "/mnt/hdd-nfs/dataset/ImageNet"
def test_all(args, name, cfg_modifier=lambda x: x, calib_size=32, config_name="PTQ4ViT"):
    # print('*init config...')

    # copy string from config_name 
    cfg = copy.deepcopy(config_name)


    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg)

    # print('*init net...')
    net = get_net(args, name)

    # print('*wrap net...')
    wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)
    
    # print('*load data...')
    train_bsz=32; valid_bsz=32; n_workers=8
    # g=datasets.ViTImageNetLoaderGenerator(PATH,'imagenet',32,32,16, kwargs={"model":net})
    
    
    g=datasets.CustomLoaderGenerator(f"/mnt/hdd-nfs/dataset/transfer/",
                                    args.dataset,
                                    train_bsz, valid_bsz, n_workers,
                                    kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=calib_size)
    # print(calib_loader)
    # check properti8se of calib loader
    # for i, (data, target) in enumerate(calib_loader):
    #     print(data.shape)
    print('data setup done.')

    # add timing
    if not args.eval:
        calib_start_time = time.time()
        print('*quantization calibrator set up...')
        quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=16) # 16 is too big for ViT-L-16 (default: 4)
        # print('*batching quant calib...')
        quant_calibrator.batching_quant_calib()
        calib_end_time = time.time()

    # print('* classification test...')
    acc = test_classification(net, test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])

    print(f"model: {name} \n")
    print(f"calibration size: {calib_size}")
    # print(f"bit settings: {quant_cfg.bit} \n")
    # print(f"config: {config_name} \n")
    # print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n")
    # print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n")
    # print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n")
    # if not args.eval:
    print(f"calibration time: {(calib_end_time-calib_start_time)/60:.2f}min")
    print(f"accuracy: {acc:.3f} \n\n")
    print("=======> ", args.resume, "||", cfg, "||", quant_cfg.bit, "<=======\n")

    print("="*80, "\n\n")

class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self,name,value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg

if __name__=='__main__':
    args = parse_args()

    names = [
        # "vit_tiny_patch16_224",
        # "vit_small_patch32_224",
        # "vit_small_patch16_224",
        "vit_base_patch16_224",
        # "vit_base_patch16_384",

        # "deit_tiny_patch16_224",
        # "deit_small_patch16_224",
        # "deit_base_patch16_224",
        # "deit_base_patch16_384",

        # "swin_tiny_patch4_window7_224",
        # "swin_small_patch4_window7_224",
        # "swin_base_patch4_window7_224",
        # "swin_base_patch4_window12_384",
        ]
    metrics = ["hessian"]
    linear_ptq_settings = [(1,1,1)] # n_V, n_H, n_a
    # calib_sizes = [32,128]
    calib_sizes = [16]
    # bit_settings = [(8,8)]
    # bit_settings = [(8,8), (6,6)] # weight, activation
    # bit_settings = [(6,16), (5,16), (4,16)]
    # bit_settings = [(4,16)]
    bit_settings = [(8,8), (6,6), (6,16), (5,16), (4,16)]

    # config_names = ["PTQ4ViT"]
    config_names = ["BasePTQ", "PTQ4ViT"]

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in product(names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names):
        cfg_list.append({
            "name": name,
            "cfg_modifier": cfg_modifier(linear_ptq_setting=linear_ptq_setting, metric=metric, bit_setting=bit_setting),
            "calib_size":calib_size,
            "config_name": config_name
        })
    
    if args.multiprocess:
        multiprocess(test_all, cfg_list, n_gpu=args.n_gpu)
    else:
        for cfg in cfg_list:
            test_all(args, **cfg)