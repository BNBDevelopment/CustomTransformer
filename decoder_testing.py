import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from decoder import Decoder


####################################
num_encoder_layers: int = 6
num_decoder_layers: int = 6
dim_model: int = 510 # Cant be 512 - 6 heads were requested, d_model must be evenly divisible by 6
num_heads: int = 6
dim_feedforward: int = 2048
dropout: float = 0.1
cross_dimensions = 10

###################################
num_epochs = 10
sequence_length = 784
batch_size = 60 #TODO 784 #AIAYN "...batched together by approximate sequence length..."

decoder = Decoder(number_of_blocks=num_decoder_layers,
            d_model=dim_model,
            num_heads=num_heads,
            d_ff=dim_feedforward,
            dropout=dropout,
            cross_dimensions=cross_dimensions)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
training_data = datasets.MNIST(root='../input/data', download=True, train=True, transform=transform)


training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

#TODO fix learning rate/ optimzer:::   page7: We used the Adam optimizer [20] with...
optimizer = optim.Adam(decoder.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

#decoder.train()
criterion = torch.nn.BCELoss()



for epoch in range(num_epochs):

    train_loss = 0
    decoder.train()

    for index, data in enumerate(training_data_loader, 0):
        optimizer.zero_grad()


        classVectors = torch.zeros(data[1].shape[0], 10)
        for index, class_value in enumerate(data[1]):
            classVectors[index][int(class_value)] = 1

        images = data[0].view(data[0].size(0), data[0].size(1), 784)

        x = decoder(classVectors)

        for image in x:
            top_k_vals = torch.topk(image, 50, dim=-1)
            pix_v = torch.multinomial(top_k_vals[0], 1)
            pix_v = pix_v * 255

        loss = criterion(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()


    #raw_output = decoder()