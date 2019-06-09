import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim

import cv2
import matplotlib.pyplot as plt

from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale

# define the network

        # self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        # self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        # self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        # self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        # self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        # self.conv7_stage1 = nn.Conv2d(512, self.k + 1, kernel_size=1)

                #net = slim.conv2d(input_image, 64, [3, 3], scope='sub_conv1')
                #net = slim.conv2d(net, 64, [3, 3], scope='sub_conv2')
                # net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool1')
                # net = slim.conv2d(net, 128, [3, 3], scope='sub_conv3')
                # net = slim.conv2d(net, 128, [3, 3], scope='sub_conv4')
                # net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool2')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv5')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv6')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv7')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv8')
                # net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool3')
                # net = slim.conv2d(net, 512, [3, 3], scope='sub_conv9')
                # net = slim.conv2d(net, 512, [3, 3], scope='sub_conv10')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv11')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv12')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv13')
                # net = slim.conv2d(net, 256, [3, 3], scope='sub_conv14')
                # self.sub_stage_img_feature = slim.conv2d(net, 128, [3, 3],
                #                                          scope='sub_stage_img_feature')

class My_Net(nn.Module):

    def __init__(self):
        super(My_Net, self).__init__()
        # 2D convolution params: input channels, output channels, kernel size, padding
        # padding = 0: no zero padding
        # padding = 1: zero pading  
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_5 = nn.Conv2d(256, 128, 1, padding=0)
        self.conv3_6 = nn.Conv2d(128, 1, 1, padding=0)


    def forward(self, x):
        # ReLU after each convolution
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        # Max pooling over a (2, 2) window
        # params max_pool_2d: input, kernel_size, stride
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv3_4(x))
        x = F.relu(self.conv3_5(x))
        x = self.conv3_6(x)
        
        return x

def init_weights(m):
    print(m)
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #m.weight.data.xavier_uniform_(1.0)
        #print(m.weight)


if __name__ == "__main__":
    net = My_Net()
    print(net)
    net.apply(init_weights)

    my_root_dir = "/Users/Michelle/Documents/Augsburg_Uni/SS 2019/Bachelorarbeit/Aufnahmen/Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_1_Adhesion"
    dataset = CellCoreDataset(my_root_dir, 
                            transform = transforms.Compose(
                                [Rescale( int(1864 / 2) ),
                                RandomRescale( 0.9, 1.1 ), 
                                RandomCrop(224)]
                            ),
                            output_reduction=4
                )
    print(len(dataset))

    #for n in range(2,100):
    #    show_training_sample(dataset[n])
    #    plt.pause(1)


    sample = dataset[0]
    img = sample['image']

    x = torch.Tensor([img]).unsqueeze(0)
    out = net(x)
    print(out.shape)
    # print(x)

    out = net(x)
    target = torch.Tensor([sample['gt_map']]).unsqueeze(0)

    # loss function
    # mean-squared error between the input and the target
    criterion = nn.MSELoss()

    plt.imshow(out.squeeze().detach().numpy())
        #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.show()  # pause a bit so that plots are updated
    plt.imshow(target.squeeze().numpy())
    plt.show()  # pause a bit so that plots are updated




    # update the weights
    # method: Stochastic Gradient Descent (SGD)
    # formula: weight = weight - learning_rate * gradient

    # create my optimizer with learning rate lr = 0.01
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train_dataset_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=16, shuffle=True,
                                                num_workers=0)
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataset_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.Tensor(data['image']).unsqueeze(1)
            targets = torch.Tensor(data['gt_map']).unsqueeze(1)

            # plt.imshow(inputs.squeeze().squeeze().detach().numpy())
            # plt.show()  # pause a bit so that plots are updated
            # plt.imshow(targets.squeeze().squeeze().numpy())
            # plt.show()  # pause a bit so that plots are updated

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            # if i % 10 == 9:
            #     plt.imshow(outputs.squeeze().squeeze().detach().numpy())
            #     plt.show()  # pause a bit so that plots are updated
            #     plt.imshow(targets.squeeze().squeeze().numpy())
            #     plt.show()  # pause a bit so that plots are updated

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, 1000 * running_loss / i))
                running_loss = 0.0

    print('Finished Training')