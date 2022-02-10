import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import json
import time

import cv2

from data import VideoDataset
import matplotlib.pyplot as plt
from models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# set path
data_path = "./dataset/"
save_model_path = "./InceptionV3/"
meta_data = "metadata.json"
# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 64   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 128
RNN_FC_dim = 126

# training parameters
k = 2             # number of target category
epochs = 120        # training epochs
batch_size = 32
learning_rate = 1e-4
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    start_time = time.time()
    print("Training start at {}".format(start_time))
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        tmp = torch.max(output, 1)
        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        end_time = time.time()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%, time elapsed: {}'.format(
            epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), 100 * step_score, end_time-start_time))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


def test(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
    return test_loss, test_score

def get_X_y(data_folder):
    max_instance = 2000
    X = []
    y = []
    y_fake = 0
    y_true = 0
    folders = os.listdir(data_folder)
    for f in folders:
        video_folder = os.path.join(data_folder, f)
        videos = os.listdir(video_folder)
        with open(os.path.join(video_folder, meta_data)) as json_file:
            label_data = json.load(json_file)
        for v in videos:
            if v.endswith('mp4'):
                if y_true > max_instance and y_fake > max_instance:
                    break
                if label_data[v]['label'] == 'FAKE':
                    y_fake += 1
                    if y_fake > max_instance:
                        continue
                    y.append(1)
                else:
                    y_true += 1
                    if y_true > max_instance:
                        continue
                    y.append(0)
                X.append(os.path.join(video_folder, v))

    return X, y


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
print("device", device)

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True,  'pin_memory': True} if use_cuda else \
    {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}

all_X, all_y = get_X_y(data_path)

# train, test split
train_list, valid_list, train_label, valid_label = train_test_split(all_X, all_y,
                                                                  test_size=0.2, shuffle=True)
# transform = transforms.Compose([transforms.Resize([res_size, res_size]),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])


# selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
num_frames = 20

train_set, valid_set = VideoDataset(train_list, train_label, num_frames, transform=transform), \
                       VideoDataset(valid_list, valid_label, num_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)


# Test set
test_folder = "./input/deepfake-detection-challenge"
test_X, test_y = get_X_y(test_folder)
test_set = VideoDataset(test_X, test_y, num_frames, transform=transform)
test_loader = data.DataLoader(test_set, **params)


# Create model
cnn_encoder = InceptV3Encoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
#cnn_encoder = MesoInception4_v().to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)



# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    # crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
    #               list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
    #               list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())
    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
else:
    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())


optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)
#optimizer = torch.optim.RMSprop(crnn_params, lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # Test
    epoch_test2_loss, epoch_test2_score = test([cnn_encoder, rnn_decoder], device, optimizer, test_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_UCF101_ResNetCRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()
