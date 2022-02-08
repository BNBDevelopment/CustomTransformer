import PIL
import numpy
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
batch_size = 10 #TODO 784 #AIAYN "...batched together by approximate sequence length..."

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
#criterion = torch.nn.BCELoss()
#criterion = torch.nn.MSELoss(reduction='sum')
criterion = torch.nn.MSELoss()

def scaleImage(array_vals):
    min_v = numpy.min(array_vals)
    max_v = numpy.max(array_vals)

    array_vals = array_vals - min_v
    array_vals = array_vals / (max_v - min_v)

    return array_vals * 255

for epoch in range(num_epochs):
    print("Epoch: " + str(epoch))
    train_loss = 0
    decoder.train()

    for batch_number, data in enumerate(training_data_loader, 0):
        print("Batch: " + str(batch_number))
        optimizer.zero_grad()

        classVectors = torch.zeros(data[1].shape[0], 10)
        for index, class_value in enumerate(data[1]):
            classVectors[index][int(class_value)] = 1

        images = data[0].view(data[0].size(0), data[0].size(1), 784)
        images += 1

        x = decoder(classVectors)

        matching_out = None
        for generated_image in x:
            top_k_vals = torch.topk(generated_image, 50, dim=-1)
            torch.multinomial(top_k_vals[0], 1)
            pix_val = torch.gather(top_k_vals[0], 1, torch.multinomial(top_k_vals[0], 1))#top_k_vals[torch.multinomial(top_k_vals[0], 1)]
            #norm_img = torch.nn.functional.normalize(pix_val, dim=0)

            if matching_out == None:
                #matching_out = norm_img.transpose(0,-1)
                matching_out = pix_val.transpose(0, -1)
            else:
                #matching_out = torch.cat((matching_out, norm_img.transpose(0,-1)), dim=0)
                matching_out = torch.cat((matching_out, pix_val.transpose(0, -1)), dim=0)

        norm_generated = torch.nn.functional.normalize(matching_out, dim=-1).unsqueeze(1)

        norm_images = torch.nn.functional.normalize(images, dim=-1)

        loss = criterion(norm_generated, norm_images)
        loss.backward()

        #train_loss += loss.data[0]
        print("loss: " + str(str(loss.data)))
        optimizer.step()


        if batch_number % 10 == 0:
            img_file = PIL.Image.fromarray(scaleImage(norm_generated[0].view(28,28).detach().numpy()))
            img_file.save("out\\dec\\batch_" + str(batch_number) + "_num" +"0.gif")
            img_file = PIL.Image.fromarray(scaleImage(norm_generated[1].view(28,28).detach().numpy()))
            img_file.save("out\\dec\\batch_" + str(batch_number) + "_num" + "1.gif")
            img_file = PIL.Image.fromarray(scaleImage(norm_generated[2].view(28,28).detach().numpy()))
            img_file.save("out\\dec\\batch_" + str(batch_number) + "_num" + "2.gif")





    #raw_output = decoder()