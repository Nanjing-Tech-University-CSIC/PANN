import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
import copy
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import time
from models.models import ConvLSTM, SST_Phy, PANN
from skimage.metrics import structural_similarity as ssim
import argparse
from data.get_loader import split_data_by_ratio, data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data/')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--save_name', type=str, default='PANN', help='')
parser.add_argument('--current_time', default='2024', type=str)
parser.add_argument('--result_dir', default='./save/', type=str)
parser.add_argument('--modelName', default='PANN-SST', type=str, help='Model_name-Dataset_name')
args = parser.parse_args()

# load your data and replace the test data
X = np.random.rand(100, 10, 1, 64, 64)
Y = np.random.rand(100, 10, 1, 64, 64)

X = X.astype(np.float32)
Y = Y.astype(np.float32)

x_train, x_val, x_test = split_data_by_ratio(X, val_ratio=0.2, test_ratio=0.1)
y_train, y_val, y_test = split_data_by_ratio(Y, val_ratio=0.2, test_ratio=0.1)

train_loader = data_loader(x_train, y_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = data_loader(x_val, y_val, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = data_loader(x_test, y_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

def get_loss_figure(save_path, train_loss_list, val_loss_list, epoch):

    x = range(0, epoch)
    train_loss = train_loss_list
    val_loss = val_loss_list
    plt.title('Train and Val Loss Analysis')
    plt.plot(x, train_loss, color='black', label='Train')
    plt.plot(x, val_loss, color='blue', label='Val')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(save_path + "Loss_Figure")


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio):
    encoder_optimizer.zero_grad()
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    past = input_tensor
    for ei in range(input_length - 1):
        encoder_output, encoder_hidden, output_image, _, _, past = encoder(input_tensor[:, ei, :, :, :], (ei == 0), decoding=False, past=past)
        loss += criterion(output_image, input_tensor[:, ei + 1, :, :, :])

    decoder_input = input_tensor[:, -1, :, :, :]
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    past = input_tensor

    for di in range(target_length):
        decoder_output, decoder_hidden, output_image, _, _, past = encoder(decoder_input, decoding=True, past=past)
        target = target_tensor[:, di, :, :, :]
        loss += criterion(output_image, target)
        if use_teacher_forcing:
            decoder_input = target
        else:
            decoder_input = output_image

    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length


def trainIters(encoder, nepochs):
    train_losses = []
    val_losses = []

    best_mse = float('inf')
    best_epoch = 0

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=10, factor=0.1, verbose=True)
    criterion = nn.MSELoss().cuda()
    train_per_epoch = len(train_loader)

    for epoch in range(0, nepochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.003)

        for i, (data, target) in enumerate(train_loader):
            # ************** Debug Mode **********************
            # if i == 10:
            #     break
            # put data into cuda
            input_tensor = data.cuda()
            target_tensor = target.cuda()
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion,
                                  teacher_forcing_ratio)
            loss_epoch += loss

            if i % 100 == 0:
                print('Train Epoch {}: {}/{} '.format(epoch, i, train_per_epoch))

        train_losses.append(loss_epoch)
        print('epoch ', epoch, ' train loss ', loss_epoch, 'Training Time per Epoch ', time.time() - t0)

        # evaluate the model after training one epoch
        mse, mae, ssim = evaluate(encoder, val_loader)
        val_losses.append(mse)
        scheduler_enc.step(mse)
        # update the best model when evaluate loss have reduced
        if mse < best_mse:
            best_epoch = epoch + 1
            best_mse = mse
            best_model = copy.deepcopy(encoder.state_dict())
            print('The evaluate loss has reduced, the best model has saved!')


    # plot the loss figure
    get_loss_figure(args.result_dir, train_losses, val_losses, args.nepochs)
    torch.save(best_model, args.result_dir + "best_model_{}.pth".format(best_epoch))
    print('The last and best model have saved')
    print('Load the best model and Test it')
    encoder.load_state_dict(best_model)
    my_test(encoder, test_loader)


def evaluate(encoder, loader):
    total_mse, total_mae, total_ssim, total_bce = 0, 0, 0, 0
    t0 = time.time()
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            # ************** Debug Mode **********************
            # if i == 10:
            #     break
            # put data into cuda
            input_tensor = data.cuda()
            target_tensor = target.cuda()
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]
            past = input_tensor
            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, _, _, past = encoder(input_tensor[:, ei, :, :, :], (ei == 0), decoding=False, past=past)
            decoder_input = input_tensor[:, -1, :, :, :]
            predictions = []
            past = input_tensor
            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _, past = encoder(decoder_input, False, decoding=True, past=past)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            # take data data from cuda to cpu
            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions)
            predictions = predictions.swapaxes(0, 1)
            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0], predictions[a, b, 0], data_range=1.0) / (target.shape[0] * target.shape[1])

    print('*****Evaluate MSE MAE SSIM*****')

    print('MSE', total_mse / len(loader),
          'MAE', total_mae / len(loader),
          'SSIM', total_ssim / len(loader),
          ' Inference Time', time.time() - t0)

    return total_mse / len(loader), total_mae / len(loader), total_ssim / len(loader)


def my_test(encoder, loader):
    total_mse, total_mae, total_ssim = 0, 0, 0
    t0 = time.time()

    first_day_pred = []
    first_day_true = []

    total_pred = []
    total_true = []

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            # # ************** Debug Mode **********************
            # if i == 10:
            #     break
            input_tensor = data.cuda()
            target_tensor = target.cuda()
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            past_encoding = input_tensor
            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, _, _, past_encoding = encoder(input_tensor[:, ei, :, :, :], (ei == 0), past=past_encoding)
            decoder_input = input_tensor[:, -1, :, :, :]
            predictions = []
            past_decoding = input_tensor
            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _, past_decoding = encoder(decoder_input, False, False, past=past_decoding)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions)
            predictions = predictions.swapaxes(0, 1)

            for b in range(args.batch_size):
                total_pred.append(predictions[b])
                total_true.append(target[b])
                first_day_pred.append(predictions[0][0][0])
                first_day_true.append(target[0][0][0])

            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0,], predictions[a, b, 0,], data_range=1.0) / (target.shape[0] * target.shape[1])

    print('*****Test MSE MAE SSIM*****')

    print('MSE', total_mse / len(loader),
          'MAE', total_mae / len(loader),
          'SSIM', total_ssim / len(loader),
          'Test Time', time.time() - t0)

# make the folder for saving the results
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
args.current_time = current_time
os.makedirs(args.result_dir + args.current_time + '-' + args.modelName)
args.result_dir = './save/{}-{}/'.format(args.current_time, args.modelName)

PCM = SST_Phy(input_channels=4, output_channels=64)
SCM = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 256, 128, 64], n_layers=4, kernel_size=(3, 3), device=device)

PANN = PANN(PCM, SCM, device)

trainIters(PANN, args.nepochs)

