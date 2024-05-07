import torch
import torch.utils.tensorboard as tb
import numpy as np
import pickle
import torch
import torch.nn as nn
import numpy as np
import torchvision.ops as tv
from torch import optim
import torch.utils.tensorboard as tb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('mps') if torch.has_mps else torch.device('cpu')


class ActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm1d(11),
            torch.nn.Linear(11, 16),
            nn.Dropout(p=.05),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            nn.Dropout(p=.05),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(32, 3)

    def forward(self, x):
        f = self.network(x)
        return self.classifier(f)
'''
class ActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm1d(11),
            torch.nn.Linear(11, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 4),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(4, 3)

    def forward(self, x):
        f = self.network(x)
        return self.classifier(f)
'''
def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, ActionNet):
        #return torch.jit.save(torch.jit.script(model), 'image_agent.pt')
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'image_agent.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def train(args):
    # print("data opened")

    files = ["geoffrey1w.pkl","geoffrey2w.pkl","yoshua1w.pkl","yoshua2w.pkl","jurgen1w.pkl","jurgen2w.pkl","yann1w.pkl","yann2w.pkl","jurgen2w2.pkl","yoshua2w2.pkl","yoshua1w2.pkl"]
    train_images = []
    train_labels = []
    print("data opened")
    for file in files:
        with open(file, 'rb') as f:
            img, lbl=pickle.load(f)
        train_images.append(img)
        train_labels.append(lbl)
    train_images = torch.cat([torch.as_tensor(image) for image in train_images]).to(device)
    train_labels = torch.cat([torch.as_tensor(label) for label in train_labels]).to(device)
    print("data loaded")
    print(train_images.shape)
    print(train_labels.shape)
    # print(torch.isnan(train_images).any())
    # print(torch.isnan(train_labels).any())

    # Find NaN values
    nan_mask = torch.isnan(train_images)
    nan_indices = torch.nonzero(nan_mask, as_tuple=True)

    # Remove rows containing NaN values
    train_images = train_images[~nan_mask.any(dim=1)]

    # Find NaN values
    nan_mask = torch.isnan(train_labels)
    nan_indices = torch.nonzero(nan_mask, as_tuple=True)

    # Remove rows containing NaN values
    train_labels = train_labels[~nan_mask.any(dim=1)]

    # print(torch.isnan(train_images).any())
    # print(torch.isnan(train_labels).any())

    n_epochs = 500
    batch_size = 128
    n_trajectories = 10

    # Create the network
    action_net = ActionNet().to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(action_net.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Create the loss
    a_loss = torch.nn.MSELoss()
    s_loss = torch.nn.MSELoss()
    b_loss = torch.nn.BCEWithLogitsLoss()
    # Start training
    global_step = 0
    action_net.train().to(device)
    # logger = tb.SummaryWriter(log_dir+'/'+str(datetime.now()), flush_secs=1)

    for epoch in range(n_epochs):
        # print(epoch)
        losses = []
        losses_accel = []
        losses_steer = []
        losses_brake = []
        for iteration in range(0, len(train_images), batch_size):
            batch_ids = torch.randint(0, len(train_images), (batch_size,), device=device)
            batch_images = train_images[batch_ids].to(device)
            batch_labels = train_labels[batch_ids].to(device)
            o = action_net(batch_images)
            loss_accel = a_loss(o[:, 0], batch_labels[:, 0])
            loss_steer = s_loss(o[:, 1], batch_labels[:, 1])
            loss_brake = b_loss(o[:, 2], batch_labels[:, 2])
            # print(loss_accel.detach().cpu().numpy())
            # print(loss_steer.detach().cpu().numpy())
            # print(loss_brake.detach().cpu().numpy())
            losses_accel.append(loss_accel.detach().cpu().numpy())
            losses_steer.append(loss_steer.detach().cpu().numpy())
            losses_brake.append(loss_brake.detach().cpu().numpy())


            sum = loss_accel + loss_steer + loss_brake
            weight_accel = loss_accel / sum
            weight_steer = loss_steer / sum
            weight_brake = loss_brake / sum
            # loss_val = loss_accel * .1 + loss_steer * .6 + loss_brake * .3
            loss_val = loss_accel * weight_accel + loss_steer * weight_steer + loss_brake * weight_brake

            global_step += 1
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            losses.append(loss_val.detach().cpu().numpy())

        avg_loss = np.mean(losses)
        avg_loss_a = np.mean(losses_accel)
        avg_loss_s = np.mean(losses_steer)
        avg_loss_b = np.mean(losses_brake)
        scheduler.step(avg_loss)
        print('epoch %-3d \t loss = %0.3f \t lr = %0.6f' % (epoch, avg_loss, optimizer.param_groups[0]['lr']))
        print('loss_a %0.3f \t loss_s = %0.3f \t loss_b = %0.3f' % (avg_loss_a, avg_loss_s, avg_loss_b))
        print()
        action_net.eval()
        save_model(action_net)
    action_net.eval()
    save_model(action_net)


if __name__ == '__main__':
    import argparse
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #input(device)
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default="train")
    parser.add_argument('-e', '--epoch', default=30)
    parser.add_argument('-l', '--lrate', default=.005)
    # parser.add_argument('-sl', '--schedule_lr', action='store_true')

    args = parser.parse_args()
    train(args)
