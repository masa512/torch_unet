# Training script for pytorch Unet 
import torch
from torch import optim
from pathlib import Path
import os
from torch.utils.data import DataLoader, random_split
import numpy as np
import helper
from torch_percloss import Perceptual_loss
# Define the file path for input/output pair

_filepath = ''
indir = os.path.join(_filepath,'inputs')
gtdir = os.path.join(_filepath,'gt')
dir_checkpoint = os.path.join(_filepath,'check_points')

def train_unet(network, device, num_epochs: int = 1,batch_size: int = 1,learning_rate = 1E-4,r_train = 0.7,Perceptual_loss=False,pix_loss = True,layer_loss=False):
    # Step 1 : Load dataset

    dataset = unet_dataset(indir,outdir,xmin = -np.pi, xmax = np.pi, ymin = -np.pi, ymax = np.pi,)

    # Step 2 : Split training/val
    train_set, val_set = random_split(dataset, [dataset.__len__()*r_train, dataset.__len__()*(1-r_train)], generator=torch.Generator().manual_seed(0))

    # Step 3 : Dataloader in order to shuffle dateset in batches for training efficiency
    args = dict(batch_size=batch_size, pin_memory=True)
    trainloader = Dataloader(train_set, shuffle=True, **args)
    valloader = DataLoader(val_set, shuffle=False, drop_last=True, **args)

    # Step 4 : Setup optimizer, lr scheduler, 
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Step 5 : Begin training.
    
    # initialize zero vector of loss (len(loss)=num_epochs)
    train_loss_vector = np.zeros((num_epochs,1))
    val_loss_vector = np.zeros_like(train_loss_vector)
    for t in range(num_epochs):
        network.train() # Training mode
        epoch_loss = 0
            for batch in trainloader:# Extract one permutation of training data on the GPU
                input_batch = batch['Input'].to(device=device, dtype=torch.float32)
                gt_batch = batch['GT'].to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=False): # Not too sure what this does...
                    pred_batch = network(input_batch) # Prediction output
                    # Define three loss functions : Perceptual, pixel-wise loss, layer-wise loss
                    if Perceptual_loss:
                        criterion = Perceptual_loss()
                        epoch_loss += criterion(yhat=pred_batch,y=gt_batch,blocks=[0 0 1 0])
                    if pix_loss:
                        criterion = torch.nn.MSELoss()
                        epoch_loss += criterion(gt_batch,pred_batch)
                    if layer_loss:
                        epoch_loss += helper.layer_combined_loss(network = network,gt_batch = gt_batch,pred_batch = pred_batch)
    
                    train_loss_vector[t] = epoch_loss 
                
                # Perform backpropagation using the evaluated loss
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Validation stage
                network.eval() # evaluation mode
                for batch in valloader:
                    input_batch = batch['Input'].to(device=device, dtype=torch.float32)
                    gt_batch = batch['GT'].to(device=device, dtype=torch.float32)

                    with torch.no_grad(): # No gradient needed for evaluation (saves memory & time)
                        val_batch_pred = network()
                        criterion = torch.nn.MSELoss()
                        loss = criterion(gt_batch,input_batch)+ helper.layer_combined_loss(network,gt_batch,pred_batch)
                        val_loss_vector[t] = loss
                network.train()
        # Could be nice if we could save model every few epochs
    
    # Step 6 : Save fig for val


if __name__ == '__main__':












