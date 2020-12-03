import argparse
import datetime
import os
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import USCISIDataset
from net import BusterNet
from utils import CustomDataParallel

def get_args():
    parser = argparse.ArgumentParser('Buster Net')
    parser.add_argument('-n', '--num_workers', type=int, default=16, help='num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=4,  help='The number of images per batch among all devices')
    parser.add_argument('--num_gpus', type=int, default=1,  help='The number of gpus') # Multi gpus not support yet.
    parser.add_argument('--freeze_layers', nargs='*', default=None,
                        help='freeze layers with strategy')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'adamw\' or \'adam\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--lmdb_dir', type=str, default='./datasets/USCISI-CMFD', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, train_simi=True, train_mani=True, train_fusion=True, debug=False):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()
        self.model = model
        self.train_simi = train_simi
        self.train_mani = train_mani 
        self.train_fusion = train_fusion
        self.debug = debug

    def forward(self, imgs, gts):
        fusion_preds, mani_preds, simi_preds = self.model(imgs)
        simi_gts = (1 - gts[:, 2, :, :]).type(torch.float)
        mani_gts = gts[:, 0, :, :].type(torch.float)
        _, fusion_gts = gts.max(dim=1)

        loss = torch.zeros(3)
        if self.train_fusion:
            fusion_loss = self.ce_criterion(fusion_preds, fusion_gts)
            loss[0] = fusion_loss
        if self.train_mani:
            mani_preds = mani_preds.squeeze(1)
            mani_loss = self.bce_criterion(mani_preds, mani_gts)
            loss[1] = mani_loss
        if self.train_simi:
            simi_preds = simi_preds.squeeze(1)
            simi_loss = self.bce_criterion(simi_preds, simi_gts)
            loss[2] = simi_loss

        return loss


def train(opt):
    train_file = 'train.keys'
    val_file = 'valid.keys'
    # Train similarity network or manipulation network independently or the whole network.
    train_simi=True
    train_mani=True
    train_fusion=True

    # According to the papers, set input_size default to 256.  
    input_size = 256

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    train_set = USCISIDataset(opt.lmdb_dir, train_file, train_transform, target_transform)
    val_set = USCISIDataset(opt.lmdb_dir, val_file, val_transform, target_transform)
    
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                    #    'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                #   'collate_fn': collater,
                  'num_workers': opt.num_workers}

    training_generator = DataLoader(train_set, **training_params)
    val_generator = DataLoader(val_set, **val_params)

    model = BusterNet(image_size=input_size)

    if opt.load_weights is not None:
        try:
            # Load pretrain VGG16 in https://download.pytorch.org/models/vgg16-397923af.pth or continuing training
            if 'vgg16_bn' in opt.load_weights:
                vgg_backbone = torch.load(opt.load_weights)
                model.manipulation_net.load_state_dict(vgg_backbone, strict=False)
                model.similarity_net.load_state_dict(vgg_backbone, strict=False)
            else:
                model.load_state_dict(torch.load(opt.load_weights), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
        print(
            f'[Info] loaded weights: {os.path.basename(opt.load_weights)}')
    else:
        print('[Info] initializing weights...')
    #     init_weights(model)

    if opt.freeze_layers is not None:
        assert isinstance(opt.freeze_layers, list), "Required List string"
        def freeze_layers(m):
            classname = m.__class__.__name__ 
            for ntl in opt.freeze_layers:
                if ntl in classname:
                    for param in m.parameters():
                        param.require_grad = False 
        
        model.apply(freeze_layers)
        print('[Info] freeze layers in ', opt.freeze_layers)
    
    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, train_simi=train_simi, train_mani=train_mani, train_fusion=train_fusion)

    if opt.num_gpus > 1 and opt.batch_size // opt.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    os.makedirs(opt.saved_path, exist_ok=True)
    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = CustomDataParallel(model, opt.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)
    
    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    elif opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    last_step = 0
    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()
    
    num_iter_per_epoch = len(training_generator)
    
    try:
        for epoch in range(opt.num_epochs):

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                last_epoch = step // num_iter_per_epoch
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs, gts, _ = data

                    if opt.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        gts = gts.cuda()

                    optimizer.zero_grad()

                    fusion_loss, mani_loss, simi_loss = model(imgs, gts)
                    fusion_loss = fusion_loss.mean()
                    simi_loss = simi_loss.mean()
                    mani_loss = mani_loss.mean()

                    loss = fusion_loss + mani_loss + simi_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Fusion loss: {:.5f}. Mani loss: {:.5f}. Mini loss: {:.5f} Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, fusion_loss.item(),
                            mani_loss.item(), simi_loss.item(), loss.item()))
                    writer.add_scalar('Loss', loss, step)
                    writer.add_scalar('fusion_loss', fusion_loss, step)
                    writer.add_scalar('simi_loss', simi_loss, step)
                    writer.add_scalar('mani_loss', mani_loss, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'model_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_fusion_ls = []
                loss_simi_ls = []
                loss_mani_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs, gts, _ = data

                        if opt.num_gpus == 1:
                            imgs = imgs.cuda()
                            gts = gts.cuda()

                        fusion_loss, mani_loss, simi_loss = model(imgs, gts)
                        fusion_loss = fusion_loss.mean()
                        simi_loss = simi_loss.mean()
                        mani_loss = mani_loss.mean()

                        loss = fusion_loss + mani_loss + simi_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_fusion_ls.append(fusion_loss.item())
                        loss_simi_ls.append(simi_loss.item())
                        loss_mani_ls.append(mani_loss.item())

                fusion_loss = np.mean(loss_fusion_ls)
                simi_loss = np.mean(loss_simi_ls)
                mani_loss = np.mean(loss_mani_ls)
                loss = fusion_loss + simi_loss + mani_loss

                print(
                    'Val. Epoch: {}/{}. Fusion loss: {:1.5f}. Simi loss: {:1.5f}. Mani loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, fusion_loss, simi_loss, mani_loss, loss))
                writer.add_scalar('Val_Loss', loss, step)
                writer.add_scalar('Val_Fusion_loss', fusion_loss, step)
                writer.add_scalar('Val_Simi_loss', simi_loss, step)
                writer.add_scalar('Val_Mani_loss', mani_loss, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'model_{epoch}_{step}.pth')

                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, f'model_{epoch}_{step}.pth')
        writer.close()
    writer.close()

def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))

if __name__ == '__main__':
    opt = get_args()
    train(opt)
