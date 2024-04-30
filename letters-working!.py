from   io           import BytesIO
from   requests     import get

from tqdm import tqdm
from itertools import product
import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy        as np

from torch import nn
from torch import optim
import torchvision

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
    return torch.nn.Hardsigmoid()(input_tensor)
    #return torch.divide(torch.add( torch.nn.Hardtanh(min_val= -1, max_val=1)(input_tensor), 1.0) , 2.0) 

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
        
        self.mask_initial = torch.round(torch.normal(torch.full((kwargs["latent_dims"],), 0.5)))
        self.mask = torch.nn.Parameter(self.mask_initial, requires_grad=True)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        
        mask = v_hard_activation(self.mask)
        code = torch.mul(code,mask)
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed, mask #torch.abs(self.mask)

#  use gpu if available
device = 2 #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(latent_dims = 128, input_shape=784).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

num_repeats = 10;
try:
    datasets = np.load("circles.npy")
    labels = np.load("labels.npy")
    
except:
    datasets = []
    labels = []
    for i in tqdm(range(15, 84-15-1)):
        for j in range(15, 84-15-1):
            template = np.zeros((84,84), dtype = np.float32)
            
            circ = circle(template, (i, j), 15, 1.)
            label = np.array([i,j])
            
            for i in range(num_repeats):
                datasets.append( circ)
                labels.append(label)
    
    datasets = np.array(datasets)
    labels = np.array(labels)
    
    np.save("circles", datasets, allow_pickle = True)
    np.save("labels", labels, allow_pickle = True)
    
dataset_x = torch.from_numpy(datasets)
dataset_y = torch.from_numpy(labels)
    
train_loader = torch.utils.data.DataLoader(
    dataset_x, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
    
epochs = 30;
for epoch in range(epochs):
    loss = 0; r_loss = 0; m_loss = 0;
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
        l_reconstruct = nn.MSELoss()(outputs, batch_features)
        l_mask        = torch.nn.Threshold(l_reconstruct.item(), l_reconstruct.item())(torch.divide(torch.sum(mask), torch.numel(mask)))
        
        #train_loss = l_reconstruct * (l_reconstruct + l_mask)
        train_loss = l_reconstruct + l_mask
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        r_loss += l_reconstruct.item()
        m_loss += l_mask.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    r_loss = r_loss / len(train_loader)
    m_loss = m_loss / len(train_loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, r_loss = {:.6f}, m_loss = {:.6f}, ".format(epoch + 1, epochs, loss, r_loss, m_loss))
    
plt.imshow(dataset_x[0])
plt.savefig("example_before.png")

print(v_hard_activation((model.cpu().mask_initial.detach())).numpy())
print(v_hard_activation((model.cpu().mask.detach())).numpy())

plt.figure()
 
outputs, mask = model.cpu()(dataset_x[0].view(-1, 784).cpu())
plt.imshow(outputs.detach().reshape(84,84))
plt.savefig("example.png")