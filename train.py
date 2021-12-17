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


def train_unet(network, device, num_epochs: int = 1,batch_size: int = 1,learning_rate = 1E-4,r_train = 0.7,Perceptual_loss=False,pix_loss = True,layer_loss=False):
    # Step 1 : Load dataset

    # Define transforms to apply in dataset
    trans = transforms.Compose([
                                transforms.Resize(20),
                                transforms.ToTensor(),
                               ])

    #dataset = unet_dataset(indir,outdir,xmin = -np.pi, xmax = np.pi, ymin = -np.pi, ymax = np.pi,)
    train_dataset = MNIST(root=r"MNIST_train",train=True, download=True,transform=trans)
    test_dataset = MNIST(root=r"MNIST_test",train=False, download=True,transform=trans)

    # Step 2 : Split training/val
    #train_set, val_set = random_split(dataset, [dataset.__len__()*r_train, dataset.__len__()*(1-r_train)], generator=torch.Generator().manual_seed(0))
    train_set = train_dataset
    val_set  = test_dataset
    # Step 3 : Dataloader in order to shuffle dateset in batches for training efficiency
    trainloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(val_set, shuffle=False, drop_last=True,batch_size=batch_size)

    # Step 4 : Setup optimizer, lr scheduler, 
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1, verbose=False)

    # Step 5 : Begin training.

    #  Define a vector for validation loss
    val_loss = []

    #  Choose one image to evaluate validation over
    
    val_img = next(iter(valloader))[0]
    val_gt =  val_img
    val_img = helper.AddGaussNoise(0,.5)(val_gt)
    
    # initialize zero vector of loss (len(loss)=num_epochs)

    num_batches = train_dataset.__len__()//batch_size
    for t in range(num_epochs):
        network.train() # Training mode
        for i, batch in enumerate(trainloader):# Extract one permutation of training data on the GPU
            #input_batch = helper.AddGaussNoise(0,0.1)(batch[0].to(device=device, dtype=torch.float32))
            input_batch = helper.AddGaussNoise(0,0.1)(batch[0]).to(device=device, dtype=torch.float32)
            gt_batch = batch[0].to(device=device, dtype=torch.float32)
            pred_batch = network(input_batch) # Prediction output
            # Define three loss functions : Perceptual, pixel-wise loss, layer-wise loss
            '''
            if Perceptual_loss:
                criterion = Perceptual_loss()
                epoch_loss += criterion(yhat=pred_batch,y=gt_batch,blocks=[0 0 1 0])
            '''
            if pix_loss:
                criterion = torch.nn.MSELoss()
                epoch_loss = criterion(gt_batch,pred_batch)
            '''
            if layer_loss:
                epoch_loss += helper.layer_combined_loss(network = network,gt_batch = gt_batch,pred_batch = pred_batch)
            '''

            

            # Validation stage
            network.eval() # evaluation mode
            '''
            for batch in valloader:
                input_batch = helper.AddGaussianNoise(0,0.1)(batch[0].to(device=device, dtype=torch.float32))
                gt_batch = batch[0].to(device=device, dtype=torch.float32)

                with torch.no_grad(): # No gradient needed for evaluation (saves memory & time)
                    val_batch_pred = network(input_batch)
                    criterion = torch.nn.MSELoss()
                    val_loss_vector[t] = criterion
            '''
            pred_val = network(val_img).data
            criterion = torch.nn.MSELoss()
            loss = criterion(pred_val,val_img)
            val_loss.append(loss)

            # Perform backpropagation using the evaluated loss
            optimizer.zero_grad(set_to_none=True)
            epoch_loss.backward()
            optimizer.step()

            if i > 2000:
                break

            network.train()
            
        # Could be nice if we could save model every few epochs

    # Step 6 : Save fig for val
    plt.plot(val_loss)
    plt.savefig('val_loss.png')

    plt.subplot(131)
    plt.imshow(val_img[0,0,:,:])
    plt.subplot(132)
    plt.imshow(val_gt[0,0,:,:])
    plt.subplot(133)
    plt.imshow(pred_val[0,0,:,:])
    plt.savefig('val_pair.png')

    # Step 7 : Save model
    torch.save(network.state_dict(), 'final_model.pt')

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

    train_unet(network=network, 
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                num_epochs = 1,
                batch_size = 4,
                learning_rate = 1E-4,
                r_train = 0.7,
                Perceptual_loss=False,
                pix_loss = True,
                layer_loss=False
                )

    













