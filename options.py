import sys
sys.path.append('./')
import argparse
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--end_epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='/work3/s212645/Spectral_Reconstruction/checkpoint/cvaeSPCA/', help='path log files')
parser.add_argument("--data_root", type=str, default='/work3/s212645/Spectral_Reconstruction/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0,1', help='path log files')
parser.add_argument("--local-rank", type=int)
parser.add_argument("--multigpu", action='store_true')
parser.add_argument("--loadmodel", action='store_true')
parser.add_argument("--nonoise", action='store_true')
opt = parser.parse_args()

if __name__ == '__main__':
    print(opt.multigpu)
    print(opt.loadmodel)