'''
Author: FANG Zhiwei
Email: zwfang@link.cuhk.edu.hk/ afjsliny@gmail.com
Date: 2024-03-18 20:10:34
LastEditTime: 2024-03-18 20:17:26
Description: training the UNet model for edge detection in sim_OCT and real_OCT
'''
import torch.optim as optim
import torch.nn as nn
import torch
from UNetModel import UNet
from torch.utils.data import DataLoader

model = UNet()
num_epochs = 5
batch_size = 4

# define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        # compute loss and backword
        loss = criterion(outputs, labels)
        loss.backward()

        # update weight
        optimizer.step()

        # update loss accumulation
        running_loss += loss.item()
        
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")
    print(f"Epoch {epoch+1}, Loss: {running_loss}")
