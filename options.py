import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
parser.add_argument('--prompt_len', type=int, default=2, help='number of prompt components per prompt block. Default 2 for HW4 (rain/snow).')
parser.add_argument('--num_expert', type=int, default=1, help='top-k prompt experts used for sparse routing.')
parser.add_argument('--use_cpr', type=int, default=0, help='set 1 to enable contrastive prompt regularization (CPR).')
parser.add_argument('--neg_num', type=int, default=1, help='number of negative prompt samples per batch when CPR is enabled.')
parser.add_argument('--lambda_cpr', type=float, default=0.1, help='weight of CPR loss term.')
parser.add_argument('--cpr_margin', type=float, default=0.01, help='margin used by CPR ranking loss.')
parser.add_argument('--use_tur', type=int, default=0, help='set 1 to enable task-uncertainty regularization (TUR).')
parser.add_argument('--lambda_tur', type=float, default=0.1, help='weight of TUR regularization term. Should match scale of base loss (~0.1).')
parser.add_argument('--tur_eps', type=float, default=1e-8, help='numerical epsilon for TUR normalization.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")

options = parser.parse_args()

