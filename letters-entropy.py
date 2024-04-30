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

import torchvision.datasets as pydatasets

def distanceTo0(vector):
    
    return torch.mul(vector,  torch.abs(torch.add(vector, -1.0)));

def distanceTo0_1(vector):
    
    return torch.absolute(torch.reciprocal(torch.add(vector, -0.5)))

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
    region = 0.1
    return SuperHardSigmoid(input_tensor) #torch.nn.Hardsigmoid()(input_tensor)
    #return torch.mul(torch.nn.Hardtanh(0, region) (input_tensor), 1.0/region)
    #return torch.divide(torch.add( torch.nn.Hardtanh(min_val= -1, max_val=1)(input_tensor), 1.0) , 2.0) 
    
def SuperHardSigmoid(input_tensor):
    coeff = 20.0
    
    return torch.reciprocal(torch.add(torch.exp(torch.add(torch.mul(input_tensor, -coeff), -0.0)), 1.0))
    
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
		
		self.discriminator_hidden_layer = nn.Linear(
            in_features=kwargs["latent_dims"], out_features=kwargs["latent_dims"]
        )
        self.discriminator_output_layer = nn.Linear(
            in_features=kwargs["latent_dims"], out_features=["latent_dims"]
        )
		self.discriminator_classification_layer = nn.Linear(
            in_features=kwargs["latent_dims"], out_features=["latent_dims"]
        )
        
        self.mask_initial = torch.add(torch.mul(torch.rand((kwargs["latent_dims"],), dtype=torch.float32), 0.5), -0.05)
        self.mask = torch.nn.Parameter(self.mask_initial, requires_grad=True)
        
        self.n_reconstruct = 0; self.n_mask = 0; self.n_discrete = 0;

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
        reconstructed = torch.relu(activation)
		
		#Discrim 1:
		
		activation = self.discriminator_hidden_layer(reconstructed)
        activation = torch.relu(activation)
        activation = self.discriminator_output_layer(activation)
		code = torch.relu(code)
        
        mask = v_hard_activation(self.mask)
        code = torch.mul(code,mask)
		
		activation = self.discriminator_classification_layer(code)
		fake_class = torch.sigmoid(activation)
		
		#Discrim 2:
		
		activation = self.discriminator_hidden_layer(features)
        activation = torch.relu(activation)
        activation = self.discriminator_output_layer(activation)
		code = torch.relu(code)
        
        mask = v_hard_activation(self.mask)
        code = torch.mul(code,mask)
		
		activation = self.discriminator_classification_layer(code)
		real_class = torch.sigmoid(activation)
		
        return reconstructed, mask, fake_class, real_class

#  use gpu if available
device = 2 #torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dims  = 84
latent_dims = 100

# mean-squared error loss
criterion = nn.MSELoss()

num_repeats = 10;

dims = 2
if (dims == 2):
    
    try:
        datasets = np.load("circles2d.npy")
        labels = np.load("labels2d.npy")
    
    except:
    
        datasets = []
        labels = []
        for i in tqdm(range(15, 84-15-1)):
            for j in range(15, 84-15-1):
                template = np.zeros((84,84), dtype = np.float32)

                circ = circle(template, (i, j), 15, 1.)
                label = np.array([i,j])

                for r in range(num_repeats):
                    datasets.append( circ)
                    labels.append(label)

        datasets = np.array(datasets)
        labels = np.array(labels)
        
        np.save("circles2d", datasets, allow_pickle = True)
        np.save("labels2d", labels, allow_pickle = True)
    
elif (dims == 3):
    
    try:
        datasets = np.load("circles3d.npy")
        labels = np.load("labels3d.npy")
    
    except:
    
        datasets = []
        labels = []
        for i in tqdm(range(15, 84-15-10)):
            for j in range(15, 84-15-10):
                template = np.zeros((84,84), dtype = np.float32)

                label = np.array([i,j])

                for s in range(10, 15):

                    circ = circle(template, (i, j), s, 1.)
                    label = np.array([i,j])

                    for r in range(num_repeats):
                        datasets.append(circ)
                        labels.append(label)

        datasets = np.array(datasets)
        labels = np.array(labels)
        
        np.save("circles2d", datasets, allow_pickle = True)
        np.save("labels2d", labels, allow_pickle = True)

    
dataset_x = torch.from_numpy(datasets)
dataset_y = torch.from_numpy(labels)

dataset = TensorDataset(dataset_x, dataset_y)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

mnist = 1
if (mnist):
    
    input_dims = 28 
    dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128, shuffle=True)

model = AE(latent_dims = latent_dims, input_shape=input_dims**2).to(device)

#create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
print((model.mask_initial.detach()).numpy())
print(v_hard_activation((model.mask_initial.detach())).numpy())

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
plt.imshow(dataset[0][0].reshape(input_dims, input_dims))
plt.savefig("example_before.png")
plt.figure()

masks = torch.tensor([]).to(device)

epochs = 100; 
for epoch in range(epochs):
    index = 0;
    loss = 0; r_loss = 0; m_loss = 0;
    for batch_features in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features[0].view(-1, input_dims**2).to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs, mask = model(batch_features)
        
        l_reconstruct = nn.MSELoss()(outputs, batch_features)
        #l_discrete    = torch.mean(torch.special.entr(mask)) #torch.divide(torch.sum(distanceTo0(mask)), torch.sum(mask))
        l_mask        = torch.mean(mask)
                        
        model.n_reconstruct = 100
        model.n_mask        = 0.017
        model.n_discrete    = 0.02
          
        l_reconstruct   = l_reconstruct * model.n_reconstruct
		l_discriminator =  
        #l_discrete = l_discrete * model.n_discrete
        l_mask = l_mask * model.n_mask
		

        """
        # compute training reconstruction loss
        l_reconstruct = nn.MSELoss()(outputs, batch_features)
        l_discrete        = torch.nn.Threshold(l_reconstruct.item(), l_reconstruct.item())(torch.mean(distanceTo0(mask)) * l_reconstruct)
        l_mask        = torch.nn.Threshold(l_reconstruct.item(), l_reconstruct.item())(torch.exp(torch.mean(mask) * l_reconstruct))
        """
        
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
    
    mask_size = torch.sum(mask)
    
    masks = torch.cat((masks, mask), 0)
    
    #print(model.mask.grad)
    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, r_loss = {:.6f}, m_loss = {:.6f}, l_discrete = {:.6f}, size =  {:.6f}".format(epoch + 1, epochs, loss, r_loss, m_loss, l_discrete, mask_size))
    

print(v_hard_activation((model.cpu().mask_initial.detach())).numpy())
print(v_hard_activation((model.cpu().mask.detach())).numpy())

plt.imshow(masks.cpu().detach().reshape(epochs, latent_dims))
plt.savefig("mask.png")

plt.figure()
 
outputs, mask = model.cpu()(dataset[0][0].view(-1, input_dims**2).cpu())
plt.imshow(outputs.detach().reshape(input_dims, input_dims))
plt.savefig("example.png")