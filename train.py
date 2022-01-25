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

def train_unet(network,
                device, 
                num_epochs: int = 1,
                batch_size: int = 2, 
                accum_step: int = 50, 
                learning_rate = 1E-4,
                r_train = 0.8,
                r_val = 0.2,
                loss_functions: str = [],
                loss_weights = [],
                perc_block = [0,0,0,0],
                masked: bool = False,
                ):

    # Step 1 : Load dataset and load model to device

    network.to('cuda')
    # Save location 
    project_name = "Focus"
    dir_name = project_name + time.strftime("%Y%m%d-%H%M%S")

    # Save path for val images every epoch
    progress_dir = os.path.join(dir_name,'progress')


    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    # Define transforms to apply in dataset
    trans = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.CenterCrop(800),
                                transforms.ToTensor(),
                               ])

    # Path to the dataset
    #path_img = r"./../PLM_data"
    path_img = r"./../Hela_data"
    dataset = helper.unet_dataset(dir = path_img, xmin = -np.pi , xmax=np.pi, ymin = -np.pi , ymax=np.pi, transform=trans)
    #dataset = helper.unet_dataset_collagen(dir = path_img, xmin = -np.pi , xmax=np.pi, ymin = 0 , ymax= 20000, transform=trans)
    # Step 2 : Split training/val
    train_count = round(r_train*dataset.__len__())
    print(train_count)
    val_count = round(r_val*dataset.__len__())
    test_count = dataset.__len__()- train_count - val_count
    train_set, val_set, test_set = random_split(dataset, [train_count,val_count,test_count])

    # Step 3 : Dataloader in order to shuffle dateset in batches for training efficiency
    trainloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=1) # This one set as 1

    # Step 4 : Setup optimizer, lr scheduler, 
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)

    # Step 5 : Begin training.

    #  Define a vector for validation loss
    train_loss = []
    val_loss = []
    num_batches = train_set.__len__()//batch_size

    # Redefine the accumulation step
    accum_step = min(accum_step,round(num_batches/10))
    print(accum_step)

    # Make loss function as dictionary
    losses = {
        "perceptual loss" : percloss().to(device=device),
        "mse" : torch.nn.MSELoss(),
        "layer loss" : LayerLoss(3, device='cuda'),
        "ssim" : helper.SSIM_loss(),
        "pearson" : helper.pearson_loss()
    }
    # Define scores
    scores = {
        "psnr" : helper.PSNR_score(),
        "ssim" : helper.SSIM_score(),
        "pearson" : helper.pearson_score(),
    }


    # Training Begins
    best_val_loss = 1000 # anything high to begin with 
    epoch_loss = []
    val_loss = []
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
                _m = batch['mask'].to(device=device, dtype=torch.float32)
                y = y*_m
                gt_batch = gt_batch*_m

            end_time = time.time()
            
            loss = 0
            # When empty just do mse by default
            if not loss_functions:
                raise Exception("You can't optimize without loss function rite? ;-)")
            elif 'layer_loss' in loss_functions and masked:
                raise Exception("Layer loss can't be done with mask .... :( Use Perceptual")
            elif len(loss_functions) != len(loss_weights):
                raise Exception("Length of the loss fnc and weight used are DIFF")
            else:
                for idx, loss_fnx in enumerate(loss_functions):
                    if loss_fnx == 'layer loss':
                        loss += loss_weights[idx]*losses[loss_fnx](gt_batch,y,*intermediate)
                    elif loss == 'perceptual loss':
                        loss += loss_weights[idx]*losses[loss_fnx](gt_batch,y,blocks=perc_block)                   
                    else:
                        loss += loss_weights[idx]*losses[loss_fnx](gt_batch,y)
        
            
            loss.backward() # Accumulate gradient
            batch_losses.append(loss.item()) # Add current batch loss
            #print(f'\r loss: {epoch_loss[-1]}', end='   ')
            # If multiple of accum step -> update the parameters and zero_grad 
            # if (i+1) % accum_step == 0 or i+1 == len(trainloader):
            optimizer.step()
            optimizer.zero_grad()
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
                pred_batch_val = network(input_batch_val)# Prediction output

                # Write the validation states
                imsave(os.path.join(pred_path,batch_val['in_name'][0]),pred_batch_val[0].cpu().detach().numpy()[0,:,:,:])

                # Define three loss functions : Perceptual, pixel-wise loss, layer-wise loss
                if not loss_functions:
                    raise Exception("You can't optimize without loss function rite? ;-)")
                elif len(loss_functions) != len(loss_weights):
                    raise Exception("Length of the loss fnc and weight used are DIFF")
                else:
                    val_l = 0
                    for idx, loss_fnx in enumerate(loss_functions):
                        if loss_fnx == 'layer loss':
                            val_l += loss_weights[idx]*losses[loss_fnx](gt_batch,y,*intermediate)
                        elif loss == 'perceptual loss':
                            val_l += loss_weights[idx]*losses[loss_fnx](gt_batch,y,blocks=perc_block)
                        else:
                            val_l += loss_weights[idx]*losses[loss_fnx](gt_batch,y)
                    batch_loss.append(val_l.item())
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
            pred_img = network(input_img)# Prediction output
            in_name = batch['in_name'][0]
            gt_name = batch['gt_name'][0]

            # Evaluate Scores
            pearson_score = scores["pearson"](input_img,gt_img).item()
            psnr_score = scores["psnr"](input_img,gt_img).item()
            ssim_score = scores["ssim"](input_img,gt_img).item()

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
            imsave(os.path.join(in_path,f'image{i}.tif'),input_img.cpu().detach().numpy()[0,:,:,:])
            imsave(os.path.join(gt_path,f'image{i}.tif'),gt_img.cpu().detach().numpy()[0,:,:,:])
            imsave(os.path.join(pred_path,f'image{i}.tif'),pred_img[0].cpu().detach().numpy()[0,:,:,:])


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
        'Train Count': train_count,
        'Validation Count' : val_count,
        'Loss Functions' : loss_functions,
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

    '''
    network = UNet(decoder_probe_points=[1, 3],super_res = True)
    train_unet(network=network, 
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                num_epochs = 256,
                batch_size = 4,
                learning_rate = 4.5e-4,
                loss_functions = ['mse','pearson'],
                loss_weights = [1,1],
                #perc_block = [0,0,1,0],
                masked = True,
                )
    '''
    network = UNet(decoder_probe_points=[1, 3], super_res = False)
    train_unet(network=network,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               num_epochs=100,
               batch_size=4,
               learning_rate=4.5e-4,
               loss_functions=['mse', 'pearson', 'perceptual loss'],
               loss_weights=[1, 1, 1],
               perc_block=[0, 0, 1, 0],
               masked=True
               )

    network = UNet(decoder_probe_points=[1, 3], super_res = False)
    train_unet(network=network,
               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               num_epochs=100,
               batch_size=4,
               learning_rate=4.5e-4,
               loss_functions=['mse', 'pearson'],
               loss_weights=[1, 1],
               #perc_block=[0, 0, 1, 0],
               masked=True
               )
    












