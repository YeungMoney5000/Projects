
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

class ActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm1d(11),
            torch.nn.Linear(11, 16),
            nn.Dropout(p=.1),
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
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(32, 3)
    
    def forward(self, x):
        f = self.network(x)
        return self.classifier(f)
    
def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, ActionNet):
        return torch.jit.save(torch.jit.script(model), 'image_agent.pt')
        #return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'image_agent.pt'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def train(args):
    files = ['jVSai.pkl','jVSg.pkl','yVSj.pkl']
    train_feat = []
    train_labels = []
    print("data opened")
    for file in files:
        with open(file, 'rb') as f:
            feat, lbl=pickle.load(f)

        train_feat.append(feat)
        train_labels.append(lbl)
    train_feat = torch.cat([torch.as_tensor(feature) for feature in train_feat])
    train_labels = torch.cat([torch.as_tensor(label) for label in train_labels])
    n_epochs = 250
    batch_size = 128
    n_trajectories = 10

    # Create the network
    action_net = ActionNet().to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(action_net.parameters(),lr=.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    # Create the loss
    a_loss = torch.nn.MSELoss()   
    s_loss = torch.nn.MSELoss()   
    b_loss = torch.nn.BCEWithLogitsLoss()
    # Start training
    global_step = 0
    action_net.train().to(device)
    #logger = tb.SummaryWriter(log_dir+'/'+str(datetime.now()), flush_secs=1)

    for epoch in range(n_epochs):
        print(epoch)
        losses=[]
        for iteration in range(0, len(train_feat), batch_size):
            batch_ids = torch.randint(0, len(train_feat), (batch_size,), device=device)
            batch_images = train_feat[batch_ids].to(device)
            batch_labels = train_labels[batch_ids].to(device)
            o = action_net(batch_images)
            loss_accel = a_loss(o[:,0], batch_labels[:,0])
            loss_steer = s_loss(o[:,1], batch_labels[:,1])
            loss_brake = b_loss(o[:,2],batch_labels[:,2])

            loss_val = loss_accel*.02 + loss_steer*.9 + loss_brake*.08
            #logger.add_scalar('train/loss', loss_val, global_step)
            global_step += 1
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            #print(loss_val)
            losses.append(loss_val.detach().cpu().numpy())
        avg_loss = np.mean(losses)
        print(avg_loss)
        scheduler.step(avg_loss)
        print(optimizer.param_groups[0]['lr'])
    action_net.eval()
    save_model(action_net)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default="train")
    parser.add_argument('-e', '--epoch', default=30)
    parser.add_argument('-l', '--lrate', default=.005)
    #parser.add_argument('-sl', '--schedule_lr', action='store_true')

    args = parser.parse_args()
    train(args)
