from   io           import BytesIO
from   requests     import get

from tqdm import tqdm
from itertools import product
import torch
from torch.utils.data import TensorDataset, DataLoader

import torchvision

import matplotlib.pyplot as plt
import numpy        as np

"""
try: 
    data = np.load("dsprites")
    
except:
    url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_84x84.npz"
    download = get(url, stream = True).content
    data = np.load(BytesIO(download))

    np.save("dsprites", data, allow_pickle = True)

num_per_shape = 4096 #(245759//6*40)
squares = data['imgs'][:num_per_shape]
squares = squares.astype('float32')
labels  = data['latents_values'][:num_per_shape]
"""
def circle(img, center, radius, color):
    m,n = img.shape
    center = np.array(center, dtype = np.float32)
    x = np.arange(0, m)

    coords = product(x, x)
    coords = np.array(list(coords), dtype = np.float32)

    in_circle = np.where(np.linalg.norm(coords-center, axis = -1) < radius)[0]
    img[coords[in_circle].astype(np.uint8)[:,0], coords[in_circle].astype(np.uint8)[:,1]] = color
    
    return img

num_repeats = 1;
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
    
"""

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=184, shuffle=True, num_workers=4)
"""
dataset_x = torch.from_numpy(datasets)
dataset_y = torch.from_numpy(labels)


dataset     = TensorDataset(dataset_x, dataset_y)
loader      = DataLoader(dataset, batch_size = 16, shuffle=True)

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(21*21*64, 184),
            torch.nn.ReLU(),
            torch.nn.Linear(184, 100),
            torch.nn.Sigmoid()

        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(100, 184),
            torch.nn.ReLU(),
            torch.nn.Linear(184, 21*21*64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1, 64, 4),
            torch.nn.Sigmoid()
        )
        
        self.mask_enc = torch.nn.Sequential(
            torch.nn.Linear(84 * 84 + 10, 184),
            torch.nn.ReLU(),
            torch.nn.Linear(184, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
            torch.nn.Hardsigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        #mask = self.mask_enc(torch.cat((x, encoded), 1))
        decoded = self.decoder(encoded)#torch.mul(encoded,mask))
        return decoded #, mask

# Model Initialization
model = AE()
model = model#.cuda()
  
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss() #
  
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

epochs = 5
outputs = []
losses = []

for epoch in range(epochs):
    
    for (image, _) in tqdm(loader):
        # Reshaping the image to (-1, 784)
        
        # The gradients are set to zero,
        optimizer.zero_grad()
                  
        # Output of Autoencoder
        reconstructed = model(image.reshape(16, 1, 84, 84)) #, mask = model(image)
        
        l_reconstruct = loss_function(reconstructed, image)
        #l_mask        = torch.nn.Threshold(l_reconstruct.item(), l_reconstruct.item())(torch.divide(torch.sum(mask), torch.numel(mask)))
        
        # Calculating the loss function
        loss = l_reconstruct; # + l_reconstruct * (l_reconstruct + l_mask)
        
        #print(mask[0])
        #print(loss.item(), l_mask.item(), l_reconstruct.item())
        
        # the the gradient is computed and stored.
        # .step() performs parameter update
        loss.backward()
        optimizer.step()
        
        # Storing the losses in a list for plotting
        losses.append(loss)
        
    outputs.append((epochs, image, reconstructed))

plt.imshow(dataset[0])
plt.savefig("example_before.png")

plt.figure()
 

print(model(dataset[0]).detach().numpy().reshape([84,84]))
plt.imshow(model(dataset[0]).detach().numpy().reshape([84,84]))
plt.savefig("example.png")

print(model(dataset[1]).detach().numpy().reshape([84,84]))
plt.imshow(model(dataset[1]).detach().numpy().reshape([84,84]))
plt.savefig("example_1.png")
        
plt.figure()

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses[-100:])
plt.savefig("test_1.png")
