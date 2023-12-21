import sys
sys.path.append('./')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--gpu', type=str, default='gpua100')
parser.add_argument('--gpu_memory', type=str, default='gpu80gb')
parser.add_argument("--numgpu", type=int, default=1, help="numgpu")
parser.add_argument("--sysmem", type=int, default=40, help="sysmem")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--end_epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--gpu_id", type=str, default='0,1', help='path log files')
parser.add_argument("--multigpu", action='store_true')
parser.add_argument("--loadmodel", action='store_true')
opt = parser.parse_args()

file = 'job_queue/' + opt.model+'.sh'

scripts = ['#!/bin/sh',
            '### General options',
            '### –- specify queue --',
            '#BSUB -q '+ opt.gpu,
            '#BSUB -R "select['+ opt.gpu_memory + ']"',
            '### -- set the job Name --',
            '#BSUB -J ' + opt.model,
            '### -- ask for number of cores (default: 1) --',
            '#BSUB -n 8',
            '### -- specify that the cores must be on the same host --',
            '#BSUB -R "span[hosts=1]"',
            '### -- Select the resources: '+str(opt.numgpu)+' gpu in exclusive process mode --',
            '#BSUB -gpu "num='+str(opt.numgpu)+':mode=exclusive_process"',
            '### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now',
            '#BSUB -W 24:00',
            '# request '+str(opt.sysmem)+'GB of system-memory',
            '#BSUB -R "rusage[mem='+str(opt.sysmem)+'GB]"',
            '### -- set the email address --',
            '# please uncomment the following line and put in your e-mail address,',
            '# if you want to receive e-mail notifications on a non-default address',
            '#BSUB -u s212645@student.dtu.dk',
            '### -- send notification at start --',
            '#BSUB -B',
            '### -- send notification at completion--',
            '#BSUB -B',
            '### -- Specify the output and error file. %J is the job-id --',
            '### -- -o and -e mean append, -oo and -eo mean overwrite --',
            '#BSUB -o '+opt.model+'%J.out',
            '#BSUB -e '+opt.model+'%J.err',
            '# -- end of LSF options --']

command = [
    'nvidia-smi',
    'module load cuda/11.8',
    'module load cudnn/v8.9.1.23-prod-cuda-11.X ',
    'cd /zhome/02/b/164706/',
    'source ./miniconda3/bin/activate',
    'conda activate pytorch',
    'cd /zhome/02/b/164706/Master_Courses/2023_Fall/Spectral_Reconstruction/',
    'export PYTHONUNBUFFERED=1'
]

if opt.multigpu:
    runline = 'python -u -m torch.distributed.launch --use-env '+opt.model_path+' --multigpu --gpu_id 0,1 ' + '--batch_size ' + str(opt.batch_size)
    if opt.loadmodel:
        runline += ' --loadmodel'
    
else:
    runline = 'python -u '+opt.model_path+' --gpu_id 0 ' + '--batch_size ' + str(opt.batch_size)
    if opt.loadmodel:
        runline += ' --loadmodel'

def write(f, line):
    f.write(line + '\n')

f = open(file, 'w')
for line in scripts:
    write(f, line)
for line in command:
    write(f, line)
write(f, runline)
f.close()