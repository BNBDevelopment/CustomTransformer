from datetime import datetime

import PIL
import numpy
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from decoder import Decoder

using_device = 'cpu'
if torch.cuda.is_available():
    using_device = 'cuda'
device = torch.device(using_device)
#torch.zeros(1).cuda()

####################################
num_encoder_layers: int = 6
num_decoder_layers: int = 6
dim_model: int = 510 # Cant be 512 - 6 heads were requested, d_model must be evenly divisible by 6
num_heads: int = 6
dim_feedforward: int = 2048
dropout: float = 0.1
cross_dimensions = 32

###################################
num_epochs = 10
sequence_length = 784
batch_size = 10 #TODO 784 #AIAYN "...batched together by approximate sequence length..."
warmup_steps = 4000

decoder = Decoder(number_of_blocks=num_decoder_layers,
            d_model=dim_model,
            num_heads=num_heads,
            d_ff=dim_feedforward,
            dropout=dropout,
            cross_dimensions=cross_dimensions)

decoder#.to(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
training_data = datasets.MNIST(root='../input/data', download=True, train=True, transform=transform)


training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)



#decoder.train()
#criterion = torch.nn.BCELoss()
#criterion = torch.nn.MSELoss(reduction='sum')
#criterion = torch.nn.MSELoss()
#criterion = torch.nn.functional.cross_entropy(pred, expected)
criterion = torch.nn.CrossEntropyLoss()

def scaleImage(array_vals):
    min_v = numpy.min(array_vals)
    max_v = numpy.max(array_vals)

    array_vals = array_vals - min_v
    array_vals = array_vals / (max_v - min_v)

    return array_vals * 255


# Learning rate/ optimzer:::   page7: We used the Adam optimizer [20] with...

#Learning rate default values - see AIAYN implementation in steps below
optimizer = optim.Adam(decoder.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
losses = []
for epoch in range(num_epochs):
    epoch += 1
    print("Epoch: " + str(epoch))
    train_loss = 0
    decoder.train()

    for batch_number, data in enumerate(training_data_loader, 0):
        batch_number += 1
        print("Batch: " + str(batch_number) + "\t\t Current Time: " + str(datetime.now().strftime("%H:%M:%S")))

        #learning_rate = 0.0001
        learning_rate = dim_model ** (-.5) * min((batch_number * batch_size * epoch) ** (-.5), (batch_number * batch_size * epoch) * (warmup_steps ** (-1.5)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        print("learning rate: " + str(learning_rate))

        optimizer.zero_grad()

        train_images = data[0].squeeze(1).view(batch_size, 784)
        train_classVectors = torch.nn.functional.one_hot(data[1], 10).float()

        x = decoder(train_images, train_classVectors)

        #image_converter = torch.nn.Linear(510,1)
        #image_batch = image_converter(x).squeeze(-1)

        image_batch = None
        for generated_image in x:
            top_k_vals = torch.topk(generated_image, 50, dim=-1)
            pix_val = torch.gather(top_k_vals[0], 1, torch.multinomial(top_k_vals[0],1))  # top_k_vals[torch.multinomial(top_k_vals[0], 1)]

            if image_batch == None:
                # matching_out = norm_img.transpose(0,-1)
                image_batch = pix_val.transpose(0,-1)
            else:
                # matching_out = torch.cat((matching_out, norm_img.transpose(0,-1)), dim=0)
                image_batch = torch.cat((image_batch, pix_val.transpose(0,-1)), dim=0)

        loss = criterion(image_batch, torch.nn.functional.softmax(train_images, dim=-1))
        #loss = torch.nn.functional.cross_entropy(image_batch, train_images)
        losses.append(int(loss))
        loss.backward()

        #train_loss += loss.data[0]
        print("loss: " + str(str(loss.data)))
        optimizer.step()


        if batch_number % 10 == 0:
            img_file = PIL.Image.fromarray(scaleImage(image_batch[0].view(28,28).detach().numpy()))
            img_file.save("out\\dec\\batch_" + str(batch_number) + "_num" +"0.gif")
            img_file = PIL.Image.fromarray(scaleImage(image_batch[1].view(28,28).detach().numpy()))
            img_file.save("out\\dec\\batch_" + str(batch_number) + "_num" + "1.gif")
            img_file = PIL.Image.fromarray(scaleImage(image_batch[2].view(28,28).detach().numpy()))
            img_file.save("out\\dec\\batch_" + str(batch_number) + "_num" + "2.gif")

        #if batch_number % 50 == 0:
        #    plt.plot(losses)
        #    plt.show()




    #raw_output = decoder()