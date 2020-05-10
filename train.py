import os
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
from torch.autograd import Variable
from torch import nn
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2, Discriminator
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, GANLoss
from logger import Tacotron2Logger
from hparams import create_hparams

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))

        # Copy the hparams for reproducability
        from shutil import copyfile
        copyfile('hparams.py', os.path.join(output_directory, 'hparams.py'))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(generator, discriminator, g_opt, d_opt, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'gen_dict': generator.state_dict(),
                'discrim_dict': discriminator.state_dict(),
                'g_optimizer': g_opt.state_dict(),
                'd_optimizer': d_opt.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, noise_dim):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)

            z = Variable(torch.cuda.FloatTensor(
                np.random.normal(size = (batch_size, x[0].size(1), noise_dim))))

            y_pred = model(x, z)
            break
            #loss = criterion(y_pred, y)
            #if distributed_run:
            #    reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            #else:
            #    reduced_val_loss = loss.item()
            #val_loss += reduced_val_loss
        #val_loss = val_loss / (i + 1)
    
    val_loss = 0
    model.train()
    if rank == 0:
        #print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # Load the two models
    generator = load_model(hparams)
    discriminator = Discriminator(hparams).cuda()
    
    learning_rate = hparams.learning_rate
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay,)
    
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay,)

    gen_steps = hparams.gen_steps
    discrim_steps = hparams.discrim_steps

    if hparams.fp16_run:
        from apex import amp
        generator, g_optimizer = amp.initialize(
            generator, g_optimizer, opt_level='O2')
        discriminator, d_optimizer = amp.initialize(
            discriminator, d_optimizer, opt_level='O2')

    if hparams.distributed_run:
        generator = apply_gradient_allreduce(generator)

    #criterion = Tacotron2Loss()
    #criterion = GANLoss()
    criterion = nn.BCELoss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            generator = warm_start_model(
                checkpoint_path, generator, hparams.ignore_layers)
        else:
            generator, g_optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, generator, g_optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    generator.train()
    discriminator.train()

    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = learning_rate
                
            # Discriminator loss first
            d_optimizer.zero_grad()
            
            # Sample data
            x, y = generator.parse_batch(batch)
            text = x[0]
            mels_true = y[0]
            if hparams.add_gan_noise:
                y[0].add_(torch.Tensor(np.random.normal(size = (mels_true.size()))).cuda())

            # Generate noise variable
            z = Variable(torch.cuda.FloatTensor(
                np.random.normal(size = (hparams.batch_size, x[0].size(1), hparams.noise_dim))))
            
            for _ in range(discrim_steps):
                # Real outputs
                d_real_out = discriminator(x[0], y[0])
                # Labels
                real_label = torch.full(d_real_out.size(), 0.9).cuda()
                fake_label = torch.full(d_real_out.size(), 0).cuda()

                errD_real = criterion(d_real_out, real_label)
                
                if hparams.fp16_run:
                    with amp.scale_loss(errD_real, d_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    errD_real.backward()
                
                # Forward model
                y_pred = generator(x, z)
                y_post = y_pred[1]
                if hparams.add_gan_noise:
                    y_post.add_(torch.Tensor(np.random.normal(size = (mels_true.size()))).cuda())
                
                # Fake error
                d_fake_out = discriminator(x[0], y_post.detach())
                errD_fake = criterion(d_fake_out, fake_label)
                
                if hparams.fp16_run:
                    with amp.scale_loss(errD_fake, d_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    errD_fake.backward()

                err_D = errD_real + errD_fake
                
                # Optimization step
                d_optimizer.step()
            
            # Generator loss
            g_optimizer.zero_grad()
            
            for _ in range(gen_steps):
                g_out = discriminator(x[0], y_post)
                real_label = torch.full(g_out.size(), 1.0).cuda()
                err_G = criterion(g_out, real_label)

                if hparams.fp16_run:
                    with amp.scale_loss(err_G, g_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    err_G.backward()
                
                # Gradient norm
                if hparams.fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(g_optimizer), hparams.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        generator.parameters(), hparams.grad_clip_thresh)

                g_optimizer.step()

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                g_loss, d_loss = err_G.item(), err_D.item()

            # Train log
            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train it {}: G_loss {:.6f} D_loss {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, g_loss, d_loss, grad_norm, duration), flush = True)
                logger.log_training(
                    g_loss, d_loss, grad_norm, learning_rate, duration, iteration)
            
            # Log images during training
            if not is_overflow and (iteration % hparams.iters_per_train_log == 0):
                logger.log_images(y_true = y, y_pred = y_pred, iteration = iteration)
            
            # Validation images
            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(generator, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, hparams.noise_dim)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(generator, discriminator, g_optimizer, d_optimizer,
                                    learning_rate, iteration, checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
