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
import json
def seed_everything(SEED=42):
    """
    A function to seed all random generators with SEED
    :param SEED: the value to use to seed all random generators
    :return: None
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

seed_everything()

def data_loader_wrapper(batch_size:int = 1):
    path_img = r"./../Hela_data"

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(800),
        transforms.ToTensor(),
    ])
    csv_names = ['train_012522.csv']
    csv_names.append(csv_names[0].replace('train', 'val'))
    csv_names.append(csv_names[0].replace('train', 'test'))

    data_loaders = []
    for csv_name in csv_names:
        data_set = helper.unet_dataset(dir=path_img,
                                        xmin=-np.pi,
                                        xmax=np.pi,
                                        ymin=-np.pi,
                                        ymax=np.pi,
                                        transform=trans,
                                        csv_name = csv_name)
        if 'test' in csv_name or 'val' in csv_name:
            data_loaders.append(DataLoader(data_set, shuffle=False, drop_last=False, batch_size=1))
        else:
            data_loaders.append(DataLoader(data_set, shuffle=True, drop_last = False, batch_size = batch_size))
    return data_loaders

def train_unet(network,
                device, 
                num_epochs: int = 1,
                batch_size: int = 2, 
                accum_step: int = 50, 
                learning_rate = 1E-4,
                data_loaders = None,
                loss_used: str = [],
                loss_weights = [],
                perc_block = [0,0,0,0],
                masked: bool = False,
                run_mode: str = 'train'
                ):

    # Step 1 : Load dataset and load model to device

    global gt_batch
    network.to('cuda')
    # Save location
    project_name = "Focus"
    dir_name = project_name + time.strftime("%Y%m%d-%H%M%S")

    # Save path for val images every epoch
    progress_dir = os.path.join(dir_name,'progress')


    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    # Define transforms to apply in dataset

    # Step 3 : Dataloader in order to shuffle dateset in batches for training efficiency
    trainloader,valloader = data_loaders[0],data_loaders[1]

    # Step 4 : Setup optimizer, lr scheduler, 
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)

    # Step 5 : Begin training.

    #  Define a vector for validation loss
    train_loss = []
    val_loss = []
    num_batches = len(trainloader)//batch_size

    # Redefine the accumulation step
    accum_step = min(accum_step,round(num_batches/10))

    # Make loss function as dictionary
    #loss_functions = helper.loss_function_wrapper()
    # Define scores
    scores = helper.metric_wrapper()


    # Training Begins
    best_val_loss = 1000 # anything high to begin with 
    epoch_loss = []
    for t in range(num_epochs):
        print(f"-------------------EPOCH {t+1}-----------------------")

        network.train()  # Training mode
        batch_losses = []
        ti = time.time()
        for i, batch in enumerate(trainloader):# Extract one permutation of training data on the GPU
            print(f"Batch {i}/{len(trainloader)}", end = "\r")
            input_batch = batch['Input'].to(device=device, dtype=torch.float32)
            gt_batch = batch['GT'].to(device=device, dtype=torch.float32)
            y, intermediate = network(input_batch)# Prediction output with layer

            # MASK
            if masked:
                y, gt_batch = helper.mask_wrapper(y,gt_batch,batch['mask'].to(device=device, dtype=torch.float32))

            end_time = time.time()
            loss = helper.evaluate_loss_wrapper(y,
                                                gt_batch,
                                                intermediate,
                                                loss_used,
                                                loss_weights,
                                                masked,
                                                mode = 'train',
                                                perc_block = [0,0,1,0])

            loss.backward() # Accumulate gradient
            batch_losses.append(loss.item()) # Add current batch loss
            #print(f'\r loss: {epoch_loss[-1]}', end='   ')
            # If multiple of accum step -> update the parameters and zero_grad 
            # if (i+1) % accum_step == 0 or i+1 == len(trainloader):
            optimizer.step()
            optimizer.zero_grad()
            if run_mode is 'test':
                break
            # REMOVE WHEN NECESSARY
            if i == 196:
                break
            end_time = time.time()
            print(f'{ti - end_time}')
        epoch_loss.append(np.mean(batch_losses))

        # Path to save images for PROGRESS
        pred_path = os.path.join(progress_dir,'pred_img')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
    
        network.eval() # Change to evaluation mode when evaluating validation loss
        with torch.no_grad():
            # Evaluate epoch loss every end of epoch
            batch_loss = []
            for j, batch_val in enumerate(valloader):# Extract one permutation of training data on the GPU
            #input_batch = helper.AddGaussNoise(0,0.1)(batch[0].to(device=device, dtype=torch.float32))
                input_batch_val = batch_val['Input'].to(device=device, dtype=torch.float32)
                gt_batch_val = batch_val['GT'].to(device=device, dtype=torch.float32)
                pred_batch_val, intermediate = network(input_batch_val)# Prediction output

                # Write the validation states
                #imsave(os.path.join(pred_path,batch_val['in_name'][0]),torch.squeeze(pred_batch_val).cpu().detach().numpy())

                # Define three loss functions : Perceptual, pixel-wise loss, layer-wise loss
                val_l = helper.evaluate_loss_wrapper(pred_batch_val,
                                                     gt_batch_val,
                                                     intermediate,
                                                     loss_used,
                                                     loss_weights,
                                                     masked,
                                                     mode = 'val')
                batch_loss.append(val_l.item())
                if run_mode is 'test':
                    break
            val_loss.append(np.mean(batch_loss))
        deltat = time.time()-ti
        train_loss.append(np.mean(epoch_loss))
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

    # VALIDATION SAVE WRAPPER?


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

    # Write row and stuff for info on CSV
    fields = ['Index (i)','In name','Out name','delZ (um)','Pearson','PSNR','SSIM']
    csv_entries = []
    csv_dir = os.path.join(dir_name,'inference.csv') 

    with torch.no_grad():
        for i, batch in enumerate(valloader):
            input_img = batch['Input'].to(device=device, dtype=torch.float32)
            gt_img = batch['GT'].to(device=device, dtype=torch.float32)
            pred_img, _ = network(input_img)# Prediction output
            in_name = batch['in_name'][0]
            gt_name = batch['gt_name'][0]

            # Evaluate Scores
            pearson_score = scores["pearson"](pred_img,gt_img).item()
            psnr_score = scores["psnr"](pred_img,gt_img).item()
            ssim_score = scores["ssim"](pred_img,gt_img).item()

            # Evaluate DeltaZ
            delZ = helper.evaluate_delZ_wrapper(in_name,gt_name)

            # Append the statistics 
            csv_entries.append({"idx":str(i),
                                "in_name":in_name,
                                "gt_name":gt_name,
                                "delZ": delZ,
                                "Pearson": str(pearson_score),
                                "PSNR" : str(psnr_score),
                                "SSIM" : str(ssim_score),
            })

            # Save image
            imsave(os.path.join(in_path,'x_'+in_name),torch.squeeze(input_img).cpu().detach().numpy())
            imsave(os.path.join(gt_path,'y_'+in_name),torch.squeeze(gt_img).cpu().detach().numpy())
            imsave(os.path.join(pred_path,'p_'+in_name),torch.squeeze(pred_img).cpu().detach().numpy())
            if run_mode is 'test':
                break


    # Save statistics in CSV:
    helper.csv_write_wrapper(fields,csv_entries,csv_dir)


    # Save plots
    plt.plot(train_loss)
    plt.savefig(os.path.join(dir_name,'train_loss.png'))
    plt.close()
    plt.plot(val_loss)
    plt.savefig(os.path.join(dir_name,'val_loss.png'))
    plt.close()

    # Update the training summary to json file
    args = {
        'Save Folder': dir_name,
        'Number of Epochs': num_epochs,
        'Batch Size': batch_size,
        'accum_step': accum_step,
        'Learning Rate': learning_rate,
        'Train Count': len(trainloader.dataset),
        'Validation Count' : len(valloader.dataset),
        'Loss Functions' : loss_used,
        'Loss Weights' : loss_weights,
        'Perceptual Loss Layer' : perc_block
    }
    helper.json_write_wrapper(args = args,save_path = dir_name)
    

# MAIN FUNCTION
if __name__ == '__main__':

    if torch.cuda.is_available():
        print(f"The CUDA GPU IS USED with msg {torch.cuda.is_available()}")
    else:
        print("GPU not really working: Running with CPU")

    # Initialize dataloaders
    batch_size = 4
    data_loaders = data_loader_wrapper(batch_size= batch_size)


    # YES SUPER RES, YES PERCEPTUAL
    network = UNet(decoder_probe_points=[1, 3], super_res=True)
    train_unet(network=network,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               num_epochs=1,
               batch_size=batch_size,
               learning_rate=4.5e-4,
               loss_used=['mse', 'pearson', 'perceptual loss'],
               loss_weights=[1, 1, 1],
               data_loaders=data_loaders,
               # perc_block = [0,0,1,0],
               masked=True,
               )
    # YES SUPER RES, NO PERCEPTUAL
    network = UNet(decoder_probe_points=[1, 3],super_res = True)
    train_unet(network=network, 
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                num_epochs = 100,
                batch_size = batch_size,
                learning_rate = 4.5e-4,
                loss_used = ['mse','pearson'],
                loss_weights = [1,1],
                data_loaders = data_loaders,
                #perc_block = [0,0,1,0],
                masked = True,
                )
    # NO SUPER RES, YES PERCEPTUAL
    network = UNet(decoder_probe_points=[1, 3], super_res = False)
    train_unet(network=network,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               num_epochs=100,
               batch_size=batch_size,
               learning_rate=4.5e-4,
               loss_used=['mse', 'pearson', 'perceptual loss'],
               loss_weights=[1, 1, 1],
               data_loaders = data_loaders,
               perc_block=[0, 0, 1, 0],
               masked=True
               )

    # NO SUPER RES, NO PERCEPTUAL
    network = UNet(decoder_probe_points=[1, 3], super_res = False)
    train_unet(network=network,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               num_epochs=100,
               batch_size=batch_size,
               learning_rate=4.5e-4,
               loss_used=['mse', 'pearson'],
               loss_weights=[1, 1],
               data_loaders = data_loaders,
               #perc_block=[0, 0, 1, 0],
               masked=True
               )












