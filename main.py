import torch
# --- this import needed for protobuff issue
from torch import nn
from torch.utils.data import DataLoader

from config import load_opts, save_opts
from dataset import DemoDataset
from lwtnet import LWTNet
from trainer import DemoEvalTrainer
from utils import gpu_initializer, load_checkpoint

opts = load_opts()

def main():

    import numpy as np
    np.random.seed(1)

    gpu_initializer(opts.gpu_id)

    model = LWTNet()

    trainer = DemoEvalTrainer(model, opts)

    if opts.resume:
        load_checkpoint(opts.resume, model)

    dataset = DemoDataset(video_path=opts.input_video,
                          resize=opts.resize,
                          fps=opts.fps,
                          sample_rate=opts.sample_rate)

    dataloader = DataLoader(dataset,
                            batch_size=opts.batch_size,
                            shuffle=False,
                            num_workers=opts.n_workers)

    model.eval()
    with torch.no_grad():  
        trainer.eval(dataloader)

if __name__ == '__main__':
    main()
