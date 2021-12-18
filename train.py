# Training script for pytorch Unet 
import torch
from torch import optim
from pathlib import Path
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
import helper
from torch_percloss import Perceptual_loss
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as MNIST
from unet_main import UNet
from torch_percloss import Perceptual_loss
from tifffile.tifffile import imsave
from layer_loss import LayerLoss


def train_unet(network, device, num_epochs: int = 2,batch_size: int = 1, accum_step: int = 20, learning_rate = 1E-4,r_train = 0.8,Perceptual_loss=True,pix_loss = False,layer_loss=False):
    # Step 1 : Load dataset and load model to device
    network.to(device=device)

    # Define transforms to apply in dataset
    trans = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.CenterCrop(800),
                                transforms.ToTensor(),
                               ])

    #train_dataset = MNIST(root=r"MNIST_train",train=True, download=True,transform=trans)
    #test_dataset = MNIST(root=r"MNIST_test",train=False, download=True,transform=trans)

    # Path to the dataset
    path_img = r"/home/qli/Desktop/Masa/Hela_data/In_Focus"
    dataset = helper.unet_dataset(indir = path_img, outdir = path_img, xmin = -np.pi , xmax=np.pi, ymin = -np.pi , ymax=np.pi,transform=trans)

    # Step 2 : Split training/val
    print(dataset)
    #train_set, val_set = random_split(dataset, [dataset.__len__()*r_train, dataset.__len__()*(1-r_train)], generator=torch.Generator().manual_seed(0))
    train_set, val_set = random_split(dataset, [80,20])
    #train_set = train_dataset
    #val_set  = train_dataset
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
    
    # initialize zero vector of loss (len(loss)=num_epochs)

    num_batches = train_set.__len__()//batch_size

    # Define three loss functions : Perceptual, pixel-wise loss, layer-wise loss
    '''
    if Perceptual_loss:
        criterion = Perceptual_loss().to(device=device)
        loss = criterion(yhat=pred_batch,y=gt_batch,blocks=[0, 0, 1, 0])
    '''

    if pix_loss:
        criterion = torch.nn.MSELoss()
        loss = criterion(gt_batch,pred_batch)/accum_step

    if layer_loss:
#             loss += helper.layer_combined_loss(network = network,gt_batch = gtmid, pred_batch = ymid)
        criterion = LayerLoss(3) # 2 intermediate and 1 for output
    
    for t in range(num_epochs):
        print(f"-------------------EPOCH {t}-----------------------")
        
        for i, batch in enumerate(trainloader):# Extract one permutation of training data on the GPU
            network.train() # Training mode
            #input_batch = helper.AddGaussNoise(0,0.1)(batch[0].to(device=device, dtype=torch.float32))
            input_batch = batch['Input'].to(device=device, dtype=torch.float32)
            gt_batch = batch['GT'].to(device=device, dtype=torch.float32)
            y, intermediate = network(input_batch)# Prediction output with layer
            
            loss = criterion(gt_batch, y, *intermediate)
            
            # Backprop
#             network.eval() # evaluation mode
            '''
            for batch in valloader:
                input_batch = helper.AddGaussianNoise(0,0.1)(batch[0].to(device=device, dtype=torch.float32))
                gt_batch = batch[0].to(device=device, dtype=torch.float32)

                with torch.no_grad(): # No gradient needed for evaluation (saves memory & time)
                    val_batch_pred = network(input_batch)
                    criterion = torch.nn.MSELoss()
                    val_loss_vector[t] = criterion
            '''
            loss.backward()
            # If multiple of accum step -> update the parameters and zero_grad 
            if i % accum_step == 0 and i != 0:
                train_loss.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
            
            with torch.no_grad():
                val_l = 0
                for j, batch2 in enumerate(valloader):# Extract one permutation of training data on the GPU
                #input_batch = helper.AddGaussNoise(0,0.1)(batch[0].to(device=device, dtype=torch.float32))
                    input_batch2 = batch2['Input'].to(device=device, dtype=torch.float32)
                    gt_batch2 = batch2['GT'].to(device=device, dtype=torch.float32)
                    pred_batch2 = network(input_batch2)# Prediction output
                    # Define three loss functions : Perceptual, pixel-wise loss, layer-wise loss
                    '''
                    if Perceptual_loss:
                        criterion = Perceptual_loss().to(device=device)
                        loss = criterion(yhat=pred_batch,y=gt_batch,blocks=[0, 0, 1, 0])
                    '''

                    if pix_loss:
                        criterion = torch.nn.MSELoss()
                        loss2 = criterion(gt_batch2,pred_batch2)
                    val_l += loss2
                if i % accum_step is 0 and i is not 0:
                    val_loss.append(val_l/20)       
                if i % 10 is 0:
                    print(f"epoch:{t}, iter:{i} : Train Loss = {loss}, Val Loss = {val_l}")

            network.train()
            
        # Could be nice if we could save model every few epochs

    # Step 7 : Save model
    torch.save(network.state_dict(), 'final_model.pt')

    #================================== Step 8 : Save validation outputs accordingly ==================================#
    network.eval()

    # Path to save images 
    out_path = 'val_outputs'
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

    test_loss = []
    with torch.no_grad():
        for i, batch in enumerate(valloader):
            input_batch = batch['Input'].to(device=device, dtype=torch.float32)
            gt_batch = batch['GT'].to(device=device, dtype=torch.float32)
            pred_batch = network(input_batch)# Prediction output
            '''
            if Perceptual_loss:
                    criterion = Perceptual_loss().to(device=device)
                    loss = criterion(yhat=pred_batch,y=gt_batch,blocks=[0, 0, 1, 0])
            '''
            if pix_loss:
                criterion = torch.nn.MSELoss()
                loss = criterion(gt_batch,pred_batch)
            '''
            
            if layer_loss:
                loss += helper.layer_combined_loss(network = network,gt_batch = gt_batch,pred_batch = pred_batch)
            '''
            test_loss.append(loss)
            # Save image
            imsave(os.path.join(in_path,f'image{i}.tif'),input_batch.cpu().detach().numpy())
            imsave(os.path.join(gt_path,f'image{i}.tif'),gt_batch.cpu().detach().numpy())
            imsave(os.path.join(pred_path,f'image{i}.tif'),pred_batch.cpu().detach().numpy())

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(["train","val"])
    plt.savefig('val_loss.png')
        
        


    #======================Step 8 : Save validation outputs accordingly==================================
    network.eval()

    # Path to save images 
    out_path = 'val_outputs'
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

    test_loss = []
    with torch.no_grad():
        for i, batch in enumerate(valloader):
            input_batch = batch['Input'].to(device=device, dtype=torch.float32)
            gt_batch = batch['GT'].to(device=device, dtype=torch.float32)
            pred_batch = network(input_batch)# Prediction output
            '''
            if Perceptual_loss:
                    criterion = Perceptual_loss().to(device=device)
                    loss = criterion(yhat=pred_batch,y=gt_batch,blocks=[0, 0, 1, 0])
            '''
            if pix_loss:
                criterion = torch.nn.MSELoss()
                loss = criterion(gt_batch,pred_batch)
            '''
            if layer_loss:
                loss += helper.layer_combined_loss(network = network,gt_batch = gt_batch,pred_batch = pred_batch)
            '''
            test_loss.append(loss)
            print(input_batch.cpu().detach().numpy())
            # Save image
            imsave(os.path.join(in_path,'fuck.tif'),input_batch.cpu().detach().numpy())
            imsave(os.path.join(gt_path,'fuck1.tif'),gt_batch.cpu().detach().numpy())
            imsave(os.path.join(pred_path,'fuck2.tif'),pred_batch.cpu().detach().numpy())
        
        
        


if __name__ == '__main__':
    '''
    # Define the file path for input/output pair
    _filepath = ''
    indir = os.path.join(_filepath,'inputs')
    gtdir = os.path.join(_filepath,'gt')
    dir_checkpoint = os.path.join(_filepath,'check_points')
    '''
    '''
    network = Unet( 
                 in_channels = 1, 
                 out_channels = 1, 
                 base_num_filter=32, 
                 decoder_probe_points = None 
                 )
    '''
    network = UNet()

    if torch.cuda.is_available():
        print(f"The CUDA GPU IS USED with msg {torch.cuda.is_available()}")
    else:
        print("Not really working: Running with CPU")
    train_unet(network=network, 
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                num_epochs = 50,
                batch_size = 1,
                learning_rate = 1E-4,
                r_train = 0.8,
                Perceptual_loss=False,
                pix_loss = True,
                layer_loss=False
                )

    













