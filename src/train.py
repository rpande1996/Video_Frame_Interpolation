from model import Net, Discriminator
import torch
from torch.utils.data import DataLoader
from dataloader import VimeoDataset
import numpy as np
from datetime import datetime
import os
import time
import pickle


def save_stats(save_dir, exp_time, hyperparams, stats):
    save_path = os.path.join(save_dir, exp_time)
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(os.path.join(save_path, 'hyperparams.pickle')):
        with open(os.path.join(save_path, 'hyperparams.pickle'), 'wb') as handle:
            pickle.dump(hyperparams, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
    with open(os.path.join(save_path, 'stats.pickle'), 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


if __name__ == '__main__':
    num_epochs = 20
    lr = 1e-4
    batch_size = 8
    vimeo_90k_path = '../dataset/vimeo_triplet/'
    save_stats_path = '../models/details/'
    eval_every = 1
    max_num_images = None
    save_model_path = '../models/weights.pt'
    time_it = True
    time_check_every = 20
    exp_time = datetime.now().strftime("date%d%m%Ytime%H%M%S")
    hyperparams = {
        'num_epochs': num_epochs,
        'lr': lr,
        'batch_size': batch_size,
        'eval_every': eval_every,
        'max_num_images': max_num_images
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    discriminator = Discriminator()
    discriminator.weight_init(mean=0.0, std=0.02)
    discriminator = discriminator.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    mse_loss = torch.nn.MSELoss()
    mse_loss.to(device)
    bce_loss = torch.nn.BCELoss()
    bce_loss.to(device)

    train_loss = []
    val_loss = []
    current_best_val_loss = float('inf')

    print('Building train/val dataloaders...')
    seq_dir = os.path.join(vimeo_90k_path, 'sequences')
    train_txt = os.path.join(vimeo_90k_path, 'tri_trainlist.txt')
    trainset = VimeoDataset(video_dir=seq_dir, text_split=train_txt)

    n = len(trainset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Train/val dataloaders successfully built!')

    print('\nTraining...')
    for epoch in range(num_epochs):
        num_batches = 0
        train_loss_epoch = [0, 0, 0]
        
        model.train()
        discriminator.train()
        start_time = time.time()
        if max_num_images is not None:
            train_batches = int(np.ceil(float(max_num_images) / batch_size))
        else:
            train_batches = len(trainloader)
        for i in trainloader:
            first = i['first_last_frames'][0]
            last = i['first_last_frames'][1]
            mid = i['middle_frame']
            first, last, mid = first.to(device), last.to(device), mid.to(device)

            mid_recon, flow_t_0, flow_t_1, w1, w2 = model(first, last)
            
            d_optimizer.zero_grad()
            d_real_result = discriminator(first, mid, last)
            d_fake_result = discriminator(first, mid_recon.detach(), last)
            d_loss_real = 0.5 * bce_loss(d_real_result, torch.ones_like(d_real_result).to(device))
            d_loss_fake = 0.5 * bce_loss(d_fake_result, torch.zeros_like(d_fake_result).to(device))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            optimizer.zero_grad()
            d_fake_result = discriminator(first, mid_recon, last)
            loss =  0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(d_fake_result, torch.ones_like(d_fake_result).to(device))
            loss.backward()
            optimizer.step()
            train_loss_epoch[0] += loss.item()
            train_loss_epoch[1] += d_loss_real.item()
            train_loss_epoch[2] += d_loss_fake.item()
            num_batches += 1

            if max_num_images is not None:
                if num_batches == train_batches:
                    break

            if time_it:
                time_now = time.time()
                time_taken = time_now - start_time
                start_time = time_now
                if num_batches == 1 or num_batches % time_check_every == 0:
                    batches_left = train_batches - num_batches
                    print('Epoch [{} / {}] Time per batch of {}: {} seconds --> {} seconds for {} / {} batches left, train loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(
                        epoch+1, num_epochs, mid.shape[0], time_taken, time_taken * batches_left, batches_left, train_batches, loss.item(), d_loss_real.item(), d_loss_fake.item()))

        train_loss_epoch[0] /= num_batches
        train_loss_epoch[1] /= num_batches
        train_loss_epoch[2] /= num_batches
        print('Epoch [{} / {}] Train g_loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(epoch+1, num_epochs, train_loss_epoch[0], train_loss_epoch[1], train_loss_epoch[2]))

        if epoch % eval_every == 0:
            train_loss.append([0,0,0])
            val_loss.append([0,0,0])
            train_loss[-1] = train_loss_epoch

            model.eval()
            discriminator.eval()

            start_time = time.time()
            val_batches = len(valloader)

            with torch.no_grad():
                num_batches = 0
                for i in valloader:
                    first = i['first_last_frames'][0]
                    last = i['first_last_frames'][1]
                    mid = i['middle_frame']
                    first, last, mid = first.to(device), last.to(device), mid.to(device)

                    mid_recon, _, _, _, _ = model(first, last)
                    d_fake_result = discriminator(first, mid_recon, last)
                    loss = g_loss = 0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(d_fake_result, torch.ones_like(d_fake_result).to(device))
                    d_real_result = discriminator(first, mid, last)
                    d_loss_real = 0.5 * bce_loss(d_real_result, torch.ones_like(d_real_result).to(device))
                    d_loss_fake = 0.5 * bce_loss(d_fake_result, torch.zeros_like(d_fake_result).to(device))

                    val_loss[-1][0] += loss.item()
                    val_loss[-1][1] += d_loss_real.item()
                    val_loss[-1][2] += d_loss_fake.item()
                    num_batches += 1

                    if time_it:
                        time_now = time.time()
                        time_taken = time_now - start_time
                        start_time = time_now
                        if num_batches == 1 or num_batches % time_check_every == 0:
                            batches_left = val_batches - num_batches
                            print('Evaluating at Epoch [{} / {}] {} seconds for {} / {} batches of {} left'.format(epoch+1, num_epochs,
                                time_taken * batches_left, batches_left, val_batches, mid.shape[0]))

                val_loss[-1][0] /= num_batches
                val_loss[-1][1] /= num_batches
                val_loss[-1][2] /= num_batches
                print('Val g_loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(val_loss[-1][0], val_loss[-1][1], val_loss[-1][2]))

                if val_loss[-1][0] < current_best_val_loss:
                    current_best_val_loss = val_loss[-1][0]
                    torch.save(model, save_model_path)
                    print("Saved new best model!")

            stats = {
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            save_stats(save_stats_path, exp_time, hyperparams, stats)
            print("Saved stats!")
