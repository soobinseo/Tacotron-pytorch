from network import *
from data import get_dataset, DataLoader, collate_fn, get_param_size
from torch import optim
import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

def main(args):

    # Get dataset
    dataset = get_dataset()

    # Construct model
    if use_cuda:
        model = nn.DataParallel(Tacotron().cuda())
    else:
        model = Tacotron()

    # Make optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n--------model restored at step %d--------\n" % args.restore_step)

    except:
        print("\n--------Start New Training--------\n")

    # Training
    model = model.train()

    # Make checkpoint directory if not exists
    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)

    # Decide loss function
    if use_cuda:
        criterion = nn.L1Loss().cuda()
    else:
        criterion = nn.L1Loss()

    # Loss for frequency of human register
    n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=8)

        for i, data in enumerate(dataloader):

            current_step = i + args.restore_step + epoch * len(dataloader) + 1

            optimizer.zero_grad()

            # Make decoder input by concatenating [GO] Frame
            try:
                mel_input = np.concatenate((np.zeros([args.batch_size, hp.num_mels, 1], dtype=np.float32),data[2][:,:,1:]), axis=2)
            except:
                raise TypeError("not same dimension")

            if use_cuda:
                characters = Variable(torch.from_numpy(data[0]).type(torch.cuda.LongTensor), requires_grad=False).cuda()
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
                mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torch.cuda.FloatTensor), requires_grad=False).cuda()
                linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torch.cuda.FloatTensor), requires_grad=False).cuda()

            else:
                characters = Variable(torch.from_numpy(data[0]).type(torch.LongTensor), requires_grad=False)
                mel_input = Variable(torch.from_numpy(mel_input).type(torch.FloatTensor), requires_grad=False)
                mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torch.FloatTensor), requires_grad=False)
                linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torch.FloatTensor), requires_grad=False)

            # Forward
            mel_output, linear_output = model.forward(characters, mel_input)

            # Calculate loss
            mel_loss = criterion(mel_output, mel_spectrogram)
            linear_loss = torch.abs(linear_output-linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:,:n_priority_freq,:])
            loss = mel_loss + linear_loss
            loss = loss.cuda()

            start_time = time.time()

            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            time_per_step = time.time() - start_time

            if current_step % hp.log_step == 0:
                print("time per step: %.2f sec" % time_per_step)
                print("At timestep %d" % current_step)
                print("linear loss: %.4f" % linear_loss.data[0])
                print("mel loss: %.4f" % mel_loss.data[0])
                print("total loss: %.4f" % loss.data[0])

            if current_step % hp.save_step == 0:
                save_checkpoint({'model':model.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    args = parser.parse_args()
    main(args)

