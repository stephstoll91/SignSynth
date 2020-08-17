import os.path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    #input data
    parser.add_argument('--data', type=str, default='/home/steph/Documents/Chapter3/Code/pose_regressor_pytorch-master/data/new_normed_smile_data_input_train_eval_test_subset.npz', help='A path to a file containing your data (.npz).')

    #for optimizer
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate, used for ADAM.')

    #training
    parser.add_argument('--mini_batch_size', type=int, default=32, help='batch size for training.')
    parser.add_argument('--training_epochs', type=int, default=5, help='number of training epochs.')
    parser.add_argument('--test', default=False, action='store_true', help='set for testing')

    #pose classifer network
    parser.add_argument('--hidden_size', type=int, default=128, help='size of hidden layers in pose classifier.')
    parser.add_argument('--num_coeff', type=int, default=50, help='number of blending coefficients.')

    #test
    parser.add_argument('--inline_test', type=bool, default=True, help='Perform test set at the end of each epoch.')
    parser.add_argument('--one_hot_size', type=int, default=105, help='size of one hot vector.')
    parser.add_argument('--path_to_vocab', type=str, default="./vocab_classes.txt", help="a path to a text file containing your gloss vocabulary.")

    #output
    parser.add_argument('--out_model', type=str, default="./training/6/", help='where you want your model saved.')
    parser.add_argument('--out_test', type=str, default="./test/6/", help="where you want test outputs saved.")
    parser.add_argument('--out_log', type=str, default="./log/6/", help="where you want logs saved.")

    opt = parser.parse_args()
    return opt


def build_path(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)
