# Training script for pytorch Unet 
import torch
from torch import optim
from pathlib import Path
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
import helper
from torch_percloss import Perceptual_loss as percloss
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as MNIST
from unet_main import UNet
from torch_percloss import Perceptual_loss
from tifffile.tifffile import imsave
import torch.nn.functional as F
from layer_loss import LayerLoss
from tqdm import tqdm
import time 


def train_unet(network,
                device, 
                num_epochs: int = 1,
                batch_size: int = 2, 
                accum_step: int = 50, 
                learning_rate = 1E-4,
                r_train = 0.8,
                Perceptual_loss=True,
                pix_loss = False,
                layer_loss=False):

    # Step 1 : Load dataset and load model to device
    network.to('cuda')
    # Save location 
    project_name = "Focus"
    dir_name = project_name + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    # Define transforms to apply in dataset
    trans = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.CenterCrop(800),
                                transforms.ToTensor(),
                               ])

    # Path to the dataset
    path_img = r"./../Hela_data"
    dataset = helper.unet_dataset(dir = path_img, xmin = -np.pi , xmax=np.pi, ymin = -np.pi , ymax=np.pi, transform=trans)

    # Step 2 : Split training/val
    train_count = round(r_train*dataset.__len__())
    val_count = dataset.__len__()-train_count
    train_set, val_set = random_split(dataset, [train_count,val_count])

    # Step 3 : Dataloader in order to shuffle dateset in batches for training efficiency
    trainloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(val_set, shuffle=False, drop_last=True,batch_size=batch_size)

    # Step 4 : Setup optimizer, lr scheduler, 
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)

    # Step 5 : Begin training.

    #  Define a vector for validation loss
    train_loss = []
    val_loss = []
    num_batches = train_set.__len__()//batch_size

    # Make loss function as dictionary
    losses = {
        "perceptual loss" : percloss().to(device=device),
        "mse" : torch.nn.MSELoss(),
        "layer loss" : LayerLoss(3, device='cuda'),
        "ssim" : helper.SSIM_loss(),
        "pearson" : helper.pearson_loss()
    }


    # Training Begins
    best_val_loss = 1000 # anything high to begin with 
    for t in range(num_epochs):
        print(f"-------------------EPOCH {t+1}-----------------------")

        network.train()  # Training mode
        epoch_loss = []
        ti = time.time()
        for i, batch in enumerate(trainloader):# Extract one permutation of training data on the GPU
            print(f"Batch {i+1}/{trainloader.__len__()}",end='\r')
            input_batch = batch['Input'].to('cuda')
            gt_batch = batch['GT'].to('cuda')
            y, intermediate = network(input_batch)# Prediction output with layer tuple
            end_time = time.time()
            
            loss = 0
            # When empty just do mse by default
            if layer_loss:
                loss += losses['layer loss'](gt_batch, y, *intermediate) 
            if pix_loss:
                loss += 1*losses['mse'](gt_batch,y)
                loss += 1*losses['ssim'](gt_batch,y)
                loss += 4*losses['pearson'](gt_batch,y)
            if Perceptual_loss:
                loss += losses['perceptual loss'](yhat=y,y=gt_batch,blocks=[0, 1, 0, 0])
            
            loss.backward() # Accumulate gradient
            epoch_loss.append(loss.item()) # Add current batch loss
            #print(f'\r loss: {epoch_loss[-1]}', end='   ')
            # If multiple of accum step -> update the parameters and zero_grad 
            if (i+1) % accum_step == 0 or i+1 == len(trainloader):
                optimizer.step()
                optimizer.zero_grad()
    
        network.eval() # Change to evaluation mode when evaluating validation loss
        with torch.no_grad():
            # Evaluate epoch loss every end of epoch
            val_l = 0
            for j, batch_val in enumerate(valloader):# Extract one permutation of training data on the GPU
                input_batch_val = batch_val['Input'].to(device=device, dtype=torch.float32)
                gt_batch_val = batch_val['GT'].to(device=device, dtype=torch.float32)
                pred_batch_val = network(input_batch_val)# Prediction output
                loss_val = 0
                if layer_loss:
                    loss_val += losses['layer loss'](gt_batch, y, *intermediate) 
                if pix_loss:
                    loss_val += 1*losses['mse'](gt_batch,y)
                    loss_val += 1*losses['ssim'](gt_batch,y)
                    loss_val += 4*losses['pearson'](gt_batch,y)
                if Perceptual_loss:
                    loss_val += losses['perceptual loss'](yhat=y,y=gt_batch,blocks=[0, 1, 0, 0])              
                val_l += loss_val
            val_loss.append(val_l.item()/valloader.__len__())    
        deltat = time.time()-ti
        train_loss.append(sum(epoch_loss)/len(epoch_loss))    
        print(f'===> Epoch {t+1}: Train Loss -> {train_loss[-1]}')
        print(f'===> Epoch {t+1}: Validation Loss -> {val_loss[-1]}')
        print(f"====> epoch{t+1}: Time elapsed = {deltat//60}mins {deltat%60} secs")

        if (t+1)%5 == 0 and t < num_epochs-1 and val_loss[-1] < best_val_loss: # If best validation loss so far -> Save the model
            torch.save(network.state_dict(), os.path.join(dir_name,f'model_epoch{t+1}.pt'))
            best_val_loss = val_loss[-1] # Update the best validation loss 


        network.train()

        # Could be nice if we could save model every few epochs

    # Step 7 : Save model
    torch.save(network.state_dict(), os.path.join(dir_name,'final_model.pt'))

    #================================== Step 8 : Save validation outputs accordingly ==================================#
    network.eval()

    # Path to save images 
    out_path = os.path.join(dir_name,'val_outputs')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    in_path = os.path.join(out_path,'in_img')
    if not os.path.exists(in_path):
        os.makedirs(in_path)
    gt_path = os.path.join(out_path,'gt_img')
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    pred_path = os.path.join(out_path,'pred_img')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    #test_loss = []
    with torch.no_grad():
        for i, batch in enumerate(valloader):
            input_batch = batch['Input'].to(device=device, dtype=torch.float32)
            gt_batch = batch['GT'].to(device=device, dtype=torch.float32)
            pred_batch = network(input_batch)# Prediction output
            # Save image
            imsave(os.path.join(in_path,f'image{i}.tif'),input_batch.cpu().detach().numpy()[0,:,:,:])
            imsave(os.path.join(gt_path,f'image{i}.tif'),gt_batch.cpu().detach().numpy()[0,:,:,:])
            imsave(os.path.join(pred_path,f'image{i}.tif'),pred_batch[0].cpu().detach().numpy()[0,:,:,:])

    # Save plots
    plt.plot(train_loss)
    plt.savefig(os.path.join(dir_name,'train_loss.png'))
    plt.close()
    plt.plot(val_loss)
    plt.savefig(os.path.join(dir_name,'val_loss.png'))
    plt.close()

        

# MAIN FUNCTION
if __name__ == '__main__':

    network = UNet(decoder_probe_points = [1,3],do_sqNex = True)

    if torch.cuda.is_available():
        print(f"The CUDA GPU IS USED with msg {torch.cuda.is_available()}")
    else:
        print("GPU not really working: Running with CPU")
    train_unet(network=network, 
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                num_epochs = 50,
                batch_size = 5,
                learning_rate = 4.5E-4,
                r_train = 0.8,
                Perceptual_loss=True,
                pix_loss = True,
                layer_loss=False,
                loss_functions = ['mse','pearson','ssim','Perceptual_loss'],
                weights = [1,2,1,1]
                )

    













