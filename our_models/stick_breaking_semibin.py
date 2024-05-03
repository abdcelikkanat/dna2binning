import torch
import math
import time
import sys
import random
import pickle as pkl
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from functools import partial
import argparse


class StickBreakingSemiBin(torch.nn.Module):
    def __init__(self, k, lr=0.1, epoch_num=100, batch_size=1000,
                 device=torch.device("cpu"), verbose=False, seed=0):

        super(StickBreakingSemiBin, self).__init__()

        self.__seed = seed
        self.__k = k
        self.__lr = lr
        self.__epoch_num = epoch_num
        self.__batch_size = batch_size
        self.__device = device
        self.__verbose = verbose

        self.__set_seed(seed)

        self.__set_seed(seed)

        self.__letters = ['A', 'C', 'G', 'T']
        self.__kmer2id = {''.join(kmer): idx for idx, kmer in enumerate(itertools.product(self.__letters, repeat=k))}

        self.__weights0 = torch.nn.Parameter(
            2 * torch.rand(size=((4**self.__k), 512), device=self.__device) - 1, requires_grad=True
        )
        self.__bias0 = torch.nn.Parameter(
            2 * torch.rand(size=(512,), device=self.__device) - 1, requires_grad=True
        )
        self.__weights1 = torch.nn.Parameter(
            2 * torch.rand(size=(512, 256), device=self.__device) - 1, requires_grad=True
        )
        self.__bias1 = torch.nn.Parameter(
            2 * torch.rand(size=(256,), device=self.__device) - 1, requires_grad=True
        )
        self.__weights2 = torch.nn.Parameter(
            2 * torch.rand(size=(256, 128), device=self.__device) - 1, requires_grad=True
        )
        self.__bias2 = torch.nn.Parameter(
            2 * torch.rand(size=(128,), device=self.__device) - 1, requires_grad=True
        )

        self.batchnorm0 = torch.nn.BatchNorm1d(512)
        self.batchnorm1 = torch.nn.BatchNorm1d(256)

        self.__optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)
        self.__loss = []

        self.__loss_func = torch.nn.BCELoss()

    def __set_seed(self, seed=None):

        if seed is not None:
            self._seed = seed

        random.seed(self._seed)
        torch.manual_seed(self._seed)

    def single_pass(self, kmers_profile):

        input = kmers_profile / kmers_profile.sum(dim=1).view(-1, 1)

        output0 = torch.nn.functional.relu(input @ self.__weights0 + self.__bias0, inplace=True)
        output0 = self.batchnorm0(output0)
        output1 = torch.nn.functional.relu(output0 @ self.__weights1 + self.__bias1, inplace=True)
        output1 = self.batchnorm1(output1)
        # output2 = output1 @ self.__weights2

        # Standardize the output
        output5 = output1 #(output2 - output2.mean(dim=0)) / output2.std(dim=0)

        return output5

    def get_loss(self, left_kmers_profile, right_kmers_profile):

        output5_left, output5_right = self.single_pass(left_kmers_profile), self.single_pass(right_kmers_profile)
        dist = torch.norm(output5_left - output5_right, dim=1, p=2)

        return dist

    def get_last_layer(self, sequences):

        with torch.no_grad():

            kmer_profiles = []
            for sequence in sequences:

                # Get the k-mer profile
                kmer_profile = np.zeros(shape=(4 ** self.__k,), dtype=float)
                for i in range(len(sequence) - self.__k + 1):
                    kmer_id = self.__kmer2id[sequence[i:i + self.__k]]
                    kmer_profile[kmer_id] += 1

                kmer_profiles.append(kmer_profile)

            kmer_profiles = torch.from_numpy(np.asarray(kmer_profiles)).to(torch.float)

            embeddings = self.single_pass(kmer_profiles)

        return embeddings.detach().numpy()

    def get_kmer_embs(self):

        return self.__kmer_embs.detach().numpy()

    def __compute_loss(self, left_kmers, right_kmers, labels):
        '''
        p = torch.exp(-self.get_loss(left_kmers, right_kmers, labels, read_assignments)[0])
        return self.__loss_func(p, labels.to(torch.float))
        '''
        dist = self.get_loss(left_kmers, right_kmers)
        p = torch.exp(-dist)

        ll = -1 * torch.nn.functional.binary_cross_entropy(p, labels.to(torch.float), reduction="none").mean()

        return -ll

    def get_left_right_kmer_profiles(self, file_path, read_sample_size):

        # Read the sampled lines
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                # Remove the newline character and commas
                line = line.strip().replace(',', '')
                sequences.append(line)

        # Sample 'read_sample_size' lines and sort them
        if read_sample_size > 0:
            sequences = [sequences[idx] for idx in random.sample(range(len(sequences)), read_sample_size)]

        # Construct positive samples
        left_kmer_profiles, right_kmer_profiles, labels = [], [], []
        for sequence in sequences:

            current_left_kmer_profile = np.zeros(shape=(4 ** self.__k, ), dtype=float)
            current_right_kmer_profile = np.zeros(shape=(4 ** self.__k, ), dtype=float)
            for i in range(len(sequence)//2 - self.__k):
                current_left_kmer_profile[self.__kmer2id[sequence[i:i+self.__k]]] += 1
            for i in range(len(sequence)//2, len(sequence) - self.__k):
                current_right_kmer_profile[self.__kmer2id[sequence[i:i+self.__k]]] += 1

            left_kmer_profiles.append(current_left_kmer_profile)
            right_kmer_profiles.append(current_right_kmer_profile)
            labels.append(1)

        left_kmer_profiles = torch.from_numpy(np.asarray(left_kmer_profiles))
        right_kmer_profiles = torch.from_numpy(np.asarray(right_kmer_profiles))
        labels = torch.asarray(labels, dtype=torch.float)

        # Construct negative samples
        neg_left_kmer_profiles = left_kmer_profiles
        # Shuffle the negative samples
        indices = torch.randperm(len(sequences))
        neg_right_kmer_profiles = right_kmer_profiles[indices]

        # Concatenate the positive and negative samples
        left_kmer_profiles = torch.vstack((left_kmer_profiles, neg_left_kmer_profiles))
        right_kmer_profiles = torch.vstack((right_kmer_profiles, neg_right_kmer_profiles))
        labels = torch.hstack((labels, torch.zeros(len(sequences), dtype=torch.float)))

        return left_kmer_profiles.to(torch.float), right_kmer_profiles.to(torch.float), labels

    def learn(self, file_path, read_sample_size):

        left_kmer_profiles, right_kmer_profiles, labels = self.get_left_right_kmer_profiles(file_path, read_sample_size)

        for epoch in range(self.__epoch_num):

            # Shuffle data
            indices = torch.randperm(left_kmer_profiles.shape[0])
            left_kmer_profiles = left_kmer_profiles[indices]
            right_kmer_profiles = right_kmer_profiles[indices]
            labels = labels[indices]

            epoch_loss = 0
            batch_size = self.__batch_size if self.__batch_size > 0 else left_kmer_profiles.shape[0]
            for i in range(0, left_kmer_profiles.shape[0], batch_size):

                batch_left_kmer_profiles = left_kmer_profiles[i:i + batch_size]
                batch_right_kmer_profiles = right_kmer_profiles[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                if batch_labels.shape[0] != batch_size:
                    continue

                self.__optimizer.zero_grad()

                batch_loss = self.__compute_loss(batch_left_kmer_profiles, batch_right_kmer_profiles, batch_labels)
                batch_loss.backward()
                self.__optimizer.step()

                self.__optimizer.zero_grad()

                epoch_loss += batch_loss.item()

            epoch_loss /= math.ceil(left_kmer_profiles.shape[0] / batch_size)

            if self.__verbose:
                print(f"epoch: {epoch}, loss: {epoch_loss}")

            self.__loss.append(epoch_loss)

        return self.__loss

    def save(self, file_path):

        if self.__verbose:
            print(f"+ Model file is saving.")
            print(f"\t- Target path: {file_path}")

        kwargs = {
            'k': self.__k,
            'lr': self.__lr,
            'epoch_num': self.__epoch_num,
            'batch_size': self.__batch_size,
            'device': self.__device,
            'verbose': self.__verbose,
            'seed': self.__seed
        }

        torch.save([kwargs, self.state_dict()], file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--input', type=str, help='Input sequence file')
    parser.add_argument('--k', type=int, default=2, help='k value')
    parser.add_argument('--epoch', type=int, default=1000, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size (0: no batch)')
    parser.add_argument('--device', type=str, default="cpu", help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=26042024, help='Seed for random number generator')
    parser.add_argument('--w', type=int, default=2, help='Window size')
    parser.add_argument('--read_sample_size', type=int, default=8000, help='Read sample size')
    parser.add_argument('--output', type=str, help='Output file')
    args = parser.parse_args()

    model = StickBreakingSemiBin(
        k=args.k,
        lr=args.lr, epoch_num=args.epoch, batch_size=args.batch_size,
        device=torch.device(args.device), verbose=True, seed=args.seed
    )
    loss = model.learn(file_path=args.input, read_sample_size=args.read_sample_size)

    # Save the model
    model.save(args.output)

    # Save the loss
    with open(args.output + ".loss", 'w') as f:
        for l in loss:
            f.write(f"{l}\n")

    kwargs, model_state_dict = torch.load(args.output)
    new_model = StickBreakingSemiBin(**kwargs)
    new_model.load_state_dict(model_state_dict)
    '''
    sequences = []
    with open(args.input, 'r') as f:
        for line in f:
            sequence = line.strip().replace(',', '')
            sequences.append(sequence)

    embs = new_model.get_last_layer(sequences)

    plt.figure()
    for idx, start_idx in enumerate(range(0, len(embs), 1000)):
        plt.scatter(embs[start_idx:start_idx + 1000, 0], embs[start_idx:start_idx + 1000, 1], s=1)
    # plt.scatter(embs[:, 0], embs[:, 1], s=1, c=colors[idx])
    plt.show()
    '''
