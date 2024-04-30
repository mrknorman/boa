from tqdm import tqdm
from itertools import product
import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy        as np

from torch import nn
from torch import optim

def distanceTo0(vector):
    return torch.sum(torch.mul(vector,  torch.abs(torch.add(vector, -1.0))));

def circle(img, center, radius, color):
    m,n = img.shape
    center = np.array(center, dtype = np.float32)
    x = np.arange(0, m)

    coords = product(x, x)
    coords = np.array(list(coords), dtype = np.float32)

    in_circle = np.where(np.linalg.norm(coords-center, axis = -1) < radius)[0]
    img[coords[in_circle].astype(np.uint8)[:,0], coords[in_circle].astype(np.uint8)[:,1]] = color
    
    return img

def v_hard_activation(input_tensor):
    return torch.nn.Sigmoid()(input_tensor)

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=kwargs["latent_dims"]
        )
        self.encoder_output_layer = nn.Linear(
            in_features=kwargs["latent_dims"], out_features=kwargs["latent_dims"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs["latent_dims"], out_features=kwargs["latent_dims"]
        )
        self.decoder_output_layer = nn.Linear(
            in_features=kwargs["latent_dims"], out_features=kwargs["input_shape"]
        )
        
        self.mask_initial = torch.full((kwargs["latent_dims"],), 0, dtype=torch.float32)
        rank = torch.linspace(0.1, 1.0, (kwargs["latent_dims"]))
        
        self.mask = torch.nn.Parameter(self.mask_initial, requires_grad=True)
        self.rank = torch.nn.Parameter(rank, requires_grad=False)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        mask = v_hard_activation(self.mask)
        code = torch.mul(code,mask)
        code = torch.mul(code, self.rank)
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed, mask #torch.abs(self.mask)

#  use gpu if available
device = 2 #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
latent_dimensions = 10

model = AE(latent_dims = latent_dimensions, input_shape=784).to(device)
print((model.mask_initial.detach()).numpy())
print(v_hard_activation((model.mask_initial.detach())).numpy())

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# mean-squared error loss
criterion = nn.MSELoss()

num_repeats = 1;

dims = 2

if (dims == 1):
    datasets = []
    for i in tqdm(range(15, 84-15-1)):
            template = np.zeros((84,84), dtype = np.float32)
            
            circ = circle(template, (i, 15), 15, 1.)
            
            for r in range(300):
                datasets.append( circ)
    
    datasets = np.array(datasets)
    
elif (dims == 2):
    datasets = []
    for i in tqdm(range(15, 84-15-1)):
        for j in range(15, 84-15-1):
            template = np.zeros((84,84), dtype = np.float32)
            
            circ = circle(template, (i, j), 15, 1.)
            
            for r in range(10):
                datasets.append( circ)
    
    datasets = np.array(datasets)
    
elif (dims == 3):
    datasets = []
    for i in tqdm(range(15, 84-15-1)):
        for j in range(15, 84-15-1):
            template = np.zeros((84,84), dtype = np.float32)
                        
            for s in range(1, 15):
                
                circ = circle(template, (i, j), s, 1.)
                
                for r in range(1):
                    datasets.append(circ)
    
    datasets = np.array(datasets)
    
elif (dims == 4):
    datasets = []
    for i in tqdm(range(15, 84-15-1)):
        for j in range(15, 84-15-1):
            template = np.zeros((84,84), dtype = np.float32)
            
            for s in range(1, 15):
                
                label = np.array([i,j])
                
                for c in range(1, 3):

                    circ = circle(template, (i, j), s, c)
                    label = np.array([i,j])
                
                    for r in range(num_repeats):
                        datasets.append(circ)
    
    datasets = np.array(datasets)

    
dataset_x = torch.from_numpy(datasets)
    
train_loader = torch.utils.data.DataLoader(
    dataset_x, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
    
epochs = 300;
for epoch in range(epochs):
    
    loss = 0; 
    reconstruction_loss_total = 0; 
    mask_size_loss_total = 0;
    mask_entropy_loss_total = 0;
    
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs, mask = model(batch_features)
        
        # compute training reconstruction loss
        reconstruction_loss = nn.MSELoss()(outputs, batch_features)
        mask_size_loss      = torch.mean(mask * torch.linspace(1.0, 0.1, latent_dimensions).to(device))*0.01
        mask_entropy_loss   = torch.divide(distanceTo0(mask), torch.numel(mask))*0.01
        
        mask_size_loss    = torch.sqrt(torch.nn.Threshold(reconstruction_loss.item(), reconstruction_loss.item())(mask_size_loss) * reconstruction_loss)
        mask_entropy_loss = torch.sqrt(torch.nn.Threshold(reconstruction_loss.item(), reconstruction_loss.item())(mask_entropy_loss) * reconstruction_loss)

        train_loss = reconstruction_loss + mask_size_loss + mask_entropy_loss

            
        
       # train_loss = reconstruction_loss + mask_size_loss + mask_entropy_loss
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        reconstruction_loss_total += reconstruction_loss.item()
        mask_size_loss_total += mask_size_loss.item()
        mask_entropy_loss_total += mask_entropy_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    reconstruction_loss_total = reconstruction_loss_total / len(train_loader)
    mask_size_loss_total      = mask_size_loss_total      / len(train_loader)
    mask_entropy_loss_total   = mask_entropy_loss_total   / len(train_loader)
    
    mask_size = torch.sum(mask)
    
    #print(model.mask.grad)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, reconstruction_loss = {:.6f}, mask_size_loss = {:.6f}, mask_entropy_loss = {:.6f}, size =  {:.6f}".format(epoch + 1, epochs, loss, reconstruction_loss_total, mask_size_loss_total, mask_entropy_loss_total, mask_size))
    

print(v_hard_activation((model.cpu().mask_initial.detach())).numpy())
print(v_hard_activation((model.cpu().mask.detach())).numpy())


plt.imshow(v_hard_activation(model.cpu().mask.detach()).reshape(1,torch.numel(mask)))
plt.savefig("mask.png")

plt.figure()

plt.imshow(dataset_x[0])
plt.savefig("example_before.png")

plt.figure()
 
outputs, mask = model.cpu()(dataset_x[0].view(-1, 784).cpu())
plt.imshow(outputs.detach().reshape(84,84))
plt.savefig("example.png")