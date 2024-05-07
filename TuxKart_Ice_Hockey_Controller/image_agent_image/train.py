from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms
import torch
import torch.nn as nn
import numpy as np
import torchvision.ops as tv
from torch import optim
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner()
    model = model.to(device)
    print("initialize")
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    num_epochs = int(args.epoch)
    initial_learning_rate = float(args.lrate)
    #optimizer = optim.SGD(CNNmodel.parameters(), momentum = .9, lr=initial_learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=0.00001)

    #p_criterion = nn.BCEWithLogitsLoss()
    p_criterion = nn.MSELoss()
    print("start")
    batchdata = load_data("drive_data", shuffle = True, rand_flip=True,
                        color_jit = True)
    print('data loaded')
    #valid_data = load_detection_data("dense_data/valid",shuffle = False)
    i=0
    for epoch in range(0, num_epochs):
        model.train(True)
        for img, peaks in batchdata:
            model.zero_grad() 
            ff_inp = img.float()
            ff_inp,peaks = img.to(device),peaks.to(device)
            
            peak_prob= model(ff_inp) #

            p_peak = torch.sigmoid(peak_prob * (1 - 2 * peaks))
            print(peak_prob)
            #input(peaks)
            loss = (p_peak*p_criterion(peak_prob, peaks)).mean()/p_peak.mean()
            #loss = p_criterion(peak_prob,peaks)

            print(loss)
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
            i+=1
            log(train_logger,img,peaks,peak_prob,i)
        print("epoch: "+ str(epoch))

    model.eval()
    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default="train")
    parser.add_argument('-e', '--epoch', default=30)
    parser.add_argument('-l', '--lrate', default=.005)
    #parser.add_argument('-sl', '--schedule_lr', action='store_true')

    args = parser.parse_args()
    train(args)
