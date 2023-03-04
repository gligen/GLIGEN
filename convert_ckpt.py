import torch 
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str,  default=None, help="")
parser.add_argument("--new_ckpt_path", type=str,  default=None, help="")
args = parser.parse_args()


new_conv_weight = torch.zeros(320, 4+4+1, 3, 3 )

ckpt = torch.load(args.ckpt_path, map_location="cpu")

for key,value in ckpt["model"].items():
    if key == "input_blocks.0.0.weight":
        old_conv_weight = value
        new_conv_weight[:,0:4,:,:] = old_conv_weight
        ckpt["model"]["input_blocks.0.0.weight"] = new_conv_weight

save = {"model":ckpt["model"]}
torch.save(save, args.new_ckpt_path) 

