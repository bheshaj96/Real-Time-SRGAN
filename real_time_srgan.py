from argparse import ArgumentParser
from dataloader import DataLoader
import tensorflow as tf
import os
from train import RealTimeSrganTrainer
from data import DIV2K

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
parser.add_argument('--hr_size', default=384, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                            help='The number of iterations to save the tensorboard summaries and models.')

def main():
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')


    train_loader = DIV2K(scale=4,
                         downgrade='bicubic',
                         subset='train')

    ds = train_loader.dataset(batch_size=args.batch_size,
                            random_transform=True,
                            repeat_count=1)

    ds = DataLoader(args.image_dir, args.hr_size).dataset(args.batch_size) # harcoded for scale 4 now.

    # Define the directory for saving pretrainig loss tensorboard summary.
    pretrain_summary_writer = tf.summary.create_file_writer('logs/pretrain')
    train_summary_writer = tf.summary.create_file_writer('logs/train')
    
    RealTime_srgan_trainer = RealTimeSrganTrainer(args, train_summary_writer, pretrain_summary_writer) 

    RealTime_srgan_trainer.pretrain_generator(ds)

    # Run training.
    RealTime_srgan_trainer.train(ds, args.save_iter, args.epochs)


if __name__=='__main__':
    main()
