import torch
import torch.nn
import torch.optim
import numpy as np
from pose_regressor import pose_regressor
import pickle
import Utils as utils
import os
import os.path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from make_p2v_maps import make_heatmaps


class gloss2pose(object):
    def __init__(self, args):
        # subunit predictor parameters
        self.hidden_size = args.hidden_size
        self.num_coeff = args.num_coeff
        self.one_hot_size = args.one_hot_size

        # house-keeping
        self.mini_batch_size = args.mini_batch_size
        self.training_epochs = args.training_epochs
        self.inline_test = args.inline_test
        self.test = args.test
        self.keep_prob = 0.7
        if self.test:
            self.keep_prob = 1.0
        self.out_model = args.out_model
        self.out_test = args.out_test
        self.out_log = args.out_log

        # data
        self.X, self.Y, self.Xdim, self.Ydim, self.samples, self.Xmean, self.Xstd, self.Ymean, self.Ystd, self.lab, self.label_dim, self.Lmean, self.Lstd, self.labo, self.label_o_dim, self.Lomean, self.Lostd, self.L_den = self.get_data(
            args.data)
        self.vocab_path = args.path_to_vocab

        # data-index loading and data splitting
        self.rng = np.random.RandomState(23456)

        try:
            with open(self.out_model + 'inds.pkl') as f:
                self.I_train, self.I_test = pickle.load(f)

        except:
            # randomly select training set
            Ind = np.arange(self.samples)
            self.rng.shuffle(Ind)
            self.I_test = Ind[::10]
            Ind = np.delete(Ind, np.arange(0, Ind.size, 10))
            self.rng.shuffle(Ind)
            self.I_train = Ind
            with open(self.out_model + 'inds.pkl', 'wb') as f:
                pickle.dump([self.I_train, self.I_test], f)
        # batch size and epoch
        self.total_batch = int(self.samples / self.mini_batch_size)
        print("total number of batches:", self.total_batch)
        # training set and  test set
        self.num_testBatch = np.int(self.I_test.size / self.mini_batch_size)
        self.num_trainBatch = np.int(self.I_train.size / self.mini_batch_size)
        print("training batches:", self.num_trainBatch)
        print("test batches:", self.num_testBatch)

        # model
        self.model = pose_regressor(one_hot_size=self.one_hot_size, label_dim=self.label_dim, num_coeff=self.num_coeff, hidden_size=self.hidden_size, rng=self.rng, Xdim=self.Xdim, Ydim=self.Ydim, keep_prob=self.keep_prob)

        # loss
        self.loss_fn = torch.nn.MSELoss()
        # optimizer
        self.learning_rate = args.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)



        return

    def get_data(self, data_path):
        data = np.load(data_path)
        X = data['input']
        Y = data['output']
        L = data['label_in']
        L_d = L[:, :self.one_hot_size]
        Lo = data['label_out']

        Lo = Lo[:, :self.one_hot_size]

        Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
        Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)
        Lmean, Lstd = L.mean(axis=0), L.std(axis=0)
        Lomean, Lostd = Lo.mean(axis=0), Lo.std(axis=0)

        for i in range(Xstd.size):
            if (Xstd[i] == 0):
                Xstd[i] = 1
        for i in range(Ystd.size):
            if (Ystd[i] == 0):
                Ystd[i] = 1

        X = (X - Xmean) / Xstd
        Y = (Y - Ymean) / Ystd
        L = (L - Lmean) / Lstd
        Lo = (Lo - Lomean) / Lostd

        Xdim = X.shape[1]
        Ydim = Y.shape[1]
        label_dim = L.shape[1]
        label_o_dim = Lo.shape[1]
        samples = X.shape[0]

        return X, Y, Xdim, Ydim, samples, Xmean, Xstd, Ymean, Ystd, L, label_dim, Lmean, Lstd, Lo, label_o_dim, Lomean, Lostd, L_d

    def train_nn(self):
        ii = 0
        try:
            checkpoint = torch.load(self.out_model + "out_.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            e = checkpoint['epoch']
            loss = checkpoint['loss']
        except:
            print("No model found, starting training from scratch!")
            e = 0
            loss = 0

        self.optimizer.zero_grad()
        loss_mini_batch = 0

        for epoch in range(e, self.training_epochs + 1):
            b = 0
            iii = 0
            t = 0
            for i in range(self.samples):
                if b == 0:
                    self.optimizer.zero_grad()
                if b < self.mini_batch_size:
                    index_train = self.I_train[i : (i + 1)]
                    batch_xs = torch.from_numpy(self.X[index_train]).type(torch.float32)
                    batch_ys = torch.from_numpy(self.Y[index_train]).type(torch.float32)
                    batch_labels = torch.from_numpy(self.lab[index_train]).type(torch.float32)

                    # t_0 -> t_1
                    x = batch_xs.unsqueeze(-1)
                    x = x.view(1, 2, int(self.Xdim / 2))
                    pred_0 = self.model(x, batch_labels)
                    loss = self.loss_fn(pred_0, batch_ys)
                    loss.backward(retain_graph=True)
                    loss_mini_batch += loss.item()

                    # t_1 -> t_2
                    if index_train < (len(self.Y) - 1):
                        batch_ys = torch.from_numpy(self.Y[index_train + 1]).type(torch.float32)
                        batch_labels = torch.cat((batch_labels[:, :self.one_hot_size], pred_0), 1)
                        x = pred_0.unsqueeze(-1)
                        x = x.view(1, 2, int(self.Xdim / 2))
                        pred_1 = self.model(x, batch_labels)
                        loss = self.loss_fn(pred_1, batch_ys)
                        loss.backward()
                        loss_mini_batch += loss.item()

                    b += 1

                else:
                    t += 1
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    print("Epoch: %03d, Iter: %02d, Loss: %4.4f" % (epoch, i, loss_mini_batch / (self.mini_batch_size * 2)))
                    #self.test_gloss_inline("ABEND", epoch)
                    loss = loss_mini_batch
                    loss_mini_batch = 0
                    b = 0
                if self.inline_test and t >= 10000:
                    self.test_gloss_inline("ABEND", epoch)
                    t = 0
                iii += i
            # safe model at end of epoch
            ii += iii

            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss}, self.out_model + str(epoch) + "out_.pt")

            if self.inline_test:
                self.test_gloss_inline("ABEND", epoch)

        return

    def test_nn(self):
        for e in range(1, 6):
            # testing for specific glosses using all instances in the test dataset:
            # inds_npz = "/path/to/gloss/inds/GLOSS_inds.npz"
            # pr.gen_from_test_inds(inds_npz, "GLOSS", e)

            # testing for specific glosses and running up MSE: errorss = test_gloss("GLOSS", epoch, everyth sequence)
            es = []
            errorss = self.test_gloss("ERZÄHLEN", e, 50)
            es.append(errorss)
            errorss = self.test_gloss("ABEND", e, 50)
            es.append(errorss)
            errorss = self.test_gloss("ABER", e, 50)
            es.append(errorss)
            errorss = self.test_gloss("JAHR", e, 50)
            es.append(errorss)
            errorss = self.test_gloss("VORGESTERN", e, 50)
            es.append(errorss)
            ess = np.mean(np.asarray(es))
            np.savez(self.out_test + 'e_fb' + str(e) + 'error.npz', errors_mse=ess)

        return

    def test_gloss_inline(self, gloss, epoch):

        out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss + '_e' + str(epoch) + '/'
        if not os.path.exists(out):
            os.makedirs(out)
        vocab_file = open(self.vocab_path, "r")
        one_hot_index = -1
        for line in vocab_file:
            if gloss in line:
                one_hot_index = int(line.split()[0])

        if one_hot_index > -1:
            one_hot = np.zeros(self.one_hot_size)
            one_hot[one_hot_index] = 1.0

            idd = [i for i in self.I_test if all(self.L_den[i, :] == one_hot)]

            for x in range(len(idd)):
                if x % 50 == 0:
                    batch_xs = self.X[idd[x]]
                    batch_labels = self.lab[idd[x]]
                    batch_labels_out = self.labo[idd[x]]
                    batch_labels_d = np.squeeze(batch_labels)
                    batch_labels_d = batch_labels_d[:self.one_hot_size] * self.Lstd[:self.one_hot_size] + self.Lmean[
                                                                                                          :self.one_hot_size]

                    batch_labels_d_out = np.squeeze(batch_labels_out)
                    batch_labels_d_out = batch_labels_d_out[:self.one_hot_size] * self.Lostd[
                                                                                  :self.one_hot_size] + self.Lomean[
                                                                                                        :self.one_hot_size]

                    out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss + '_e' + str(epoch) + '/' + gloss + '_test' + str(x) + '/'
                    if not os.path.exists(out):
                        os.makedirs(out)

                    samples_seq = []
                    batch_xs = np.expand_dims(batch_xs, 0)
                    batch_labels = np.expand_dims(batch_labels, 0)
                    x = batch_xs.reshape(1, 2, int(self.Xdim / 2))
                    samples_0 = self.model(torch.from_numpy(x).type(torch.float32), torch.from_numpy(batch_labels).type(torch.float32))

                    samples_0d = samples_0.detach().numpy() * self.Ystd + self.Ymean
                    samples_seq.append(samples_0d)
                    samples = samples_0.detach().numpy()
                    batch_labels = np.hstack((batch_labels[0, :self.one_hot_size], samples[0, :]))
                    batch_labels = np.expand_dims(batch_labels, 0)

                    for k in range(150):
                        x = samples.reshape(1, 2, int(self.Xdim / 2))
                        samples = self.model(torch.from_numpy(x).type(torch.float32),
                                               torch.from_numpy(batch_labels).type(torch.float32))
                        samples_d = samples.detach().numpy() * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)
                        batch_labels = np.hstack((batch_labels[0, :self.one_hot_size], samples[0, :].detach().numpy()))
                        batch_labels = np.expand_dims(batch_labels, 0)
                        samples = samples.detach().numpy()

                    for j in range(0, 1):
                        for k in range(len(samples_seq)):
                            sample = samples_seq[k]
                            x = sample[j, 0:252:2]  # predicted samples
                            y = sample[j, 1:252:2]  # predicted samples

                            #fp = out + str(j) + '_' + str(k).zfill(3) + '.png'
                            #make_heatmaps(sample[j, :], fp)
                            plt.plot(x, y, 'ro')
                            plt.gca().set_xlim([700, 1300])
                            plt.gca().set_ylim([100, 900])
                            plt.gca().invert_yaxis()
                            plt.savefig(out + str(j) + '_' + str(k).zfill(3) + '.png')
                            plt.close()


    def test_gloss(self, gloss, epoch, everyth):
        try:
            checkpoint = torch.load(self.out_model + str(epoch) + "out_.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("No model found! Exiting.")
            return

        # epoch += 1
        out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss + '_e' + str(epoch) + '/'
        if not os.path.exists(out):
            os.makedirs(out)
        vocab_file = open(self.vocab_path, "r")
        one_hot_index = -1
        for line in vocab_file:
            if gloss in line:
                one_hot_index = int(line.split()[0])

        if one_hot_index > -1:
            one_hot = np.zeros(self.one_hot_size)
            one_hot[one_hot_index] = 1.0

            idd = [i for i in self.I_test if all(self.L_den[i, :] == one_hot)]
            errors_per_frame = []
            errors_per_seq = []
            gloss_error = 0.0
            errors_frame_sum = []
            for x in range(len(idd)):
                if x % everyth == 0:

                    batch_xs = self.X[idd[x]]
                    batch_ys = self.Y[idd[x]]
                    batch_labels = self.lab[idd[x]]
                    batch_labels_out = self.labo[idd[x]]
                    batch_labels_d = np.squeeze(batch_labels)
                    batch_labels_d = batch_labels_d[:self.one_hot_size] * self.Lstd[:self.one_hot_size] + self.Lmean[
                                                                                                          :self.one_hot_size]

                    batch_labels_d_out = np.squeeze(batch_labels_out)
                    batch_labels_d_out = batch_labels_d_out[:self.one_hot_size] * self.Lostd[
                                                                                  :self.one_hot_size] + self.Lomean[
                                                                                                        :self.one_hot_size]

                    out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss + '_e' + str(
                        epoch) + '/' + gloss + '_test' + str(idd[x]) + '/'
                    print(out)
                    if not os.path.exists(out):
                        os.makedirs(out)

                    out_pose = self.out_test + 'e' + str(epoch) + 'pose/' + gloss + '_e' + str(
                        epoch) + '/' + gloss + '_test' + str(idd[x]) + '/'
                    print(out_pose)

                    if not os.path.exists(out_pose):
                        os.makedirs(out_pose)

                    samples_seq = []
                    batch_xs = np.expand_dims(batch_xs, 0)
                    batch_labels = np.expand_dims(batch_labels, 0)
                    x = batch_xs.reshape(1, 2, int(self.Xdim / 2))
                    samples_0 = self.model(torch.from_numpy(x).type(torch.float32),
                                           torch.from_numpy(batch_labels).type(torch.float32))

                    samples_0d = samples_0.detach().numpy() * self.Ystd + self.Ymean
                    samples_seq.append(samples_0d)
                    samples = samples_0.detach().numpy()
                    batch_labels = np.hstack((batch_labels[0, :self.one_hot_size], samples[0, :]))
                    batch_labels = np.expand_dims(batch_labels, 0)

                    err_frame = batch_ys[0, :252] - samples[0, :252]
                    errors_per_frame.append(np.mean(np.square(np.squeeze(err_frame))))
                    for k in range(150):
                        x = samples.reshape(1, 2, int(self.Xdim / 2))
                        samples = self.model(torch.from_numpy(x).type(torch.float32),
                                             torch.from_numpy(batch_labels).type(torch.float32))
                        samples_d = samples.detach().numpy() * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)
                        batch_labels = np.hstack((batch_labels[0, :self.one_hot_size], samples[0, :].detach().numpy()))
                        batch_labels = np.expand_dims(batch_labels, 0)
                        samples = samples.detach().numpy()

                        err_frame = batch_ys[0, :252] - samples[0, :252]
                        errors_per_frame.append(np.mean(np.square(np.squeeze(err_frame))))

                    errors_frame_sum = np.mean(np.asarray(errors_per_frame))
                    errors_per_frame = []
                    errors_per_seq.append(errors_frame_sum)

                    X = []
                    Y = []

                    for j in range(0, self.mini_batch_size):
                        for k in range(len(samples_seq)):
                            sample = samples_seq[k]
                            X.append(sample[j, 0:504:2])
                            Y.append(sample[j, 1:504:2])
                            x = sample[j, 0:252:2]  # predicted samples
                            y = sample[j, 1:252:2]  # predicted samples

                            fp = out + str(j) + '_' + str(k).zfill(3) + '.png'
                            make_heatmaps(sample[j, :], fp)
                            plt.plot(x, y, 'ro')
                            plt.gca().set_xlim([700, 1300])
                            plt.gca().set_ylim([100, 900])
                            plt.gca().invert_yaxis()
                            plt.savefig(out_pose + str(j) + '_' + str(k).zfill(3) + '.png')
                            plt.close()
                    np.save(out_pose + '/Xsamples.npy', np.asarray(X))
                    np.save(out_pose + '/Ysamples.npy', np.asarray(Y))
            gloss_error = np.mean(np.asarray(errors_per_seq))
            np.savez(self.out_test + 'e' + str(epoch) + 'maps/' + gloss + '_e' + str(epoch) + '/error.npz',
                     errors_mse=gloss_error)

        vocab_file.close()
        return gloss_error

args = utils.parse_args()
utils.build_path([args.out_model])
utils.build_path([args.out_test])
utils.build_path([args.out_log])

g2p = gloss2pose(args)

if args.test:
    g2p.test_nn()
else:
    g2p.train_nn()

    # # for testing sequences across whole test set from last epoch: pr.test_sequence()
    #
    # # for specific epoch(s) and glosses:
    # for e in range(2, 3):
    #     # testing for specific glosses using all instances in the test dataset:
    #     # inds_npz = "/path/to/gloss/inds/GLOSS_inds.npz"
    #     # pr.gen_from_test_inds(inds_npz, "GLOSS", e)
    #
    #     # testing for gloss sequences: pr.test_gloss_seq("GLOSS SEQUENCE", epoch, gloss_length, everyth sequence)
    #     pr.test_gloss_seq("JETZT SPIELEN", e, 60, 100)
    #
    #     # for GT fed sequences use: pr.test_gloss_seq_GT(glosses, epoch, everyth)
    #
    #     # testing for whole vocabulary:
    #     # vocab_file = open('/path/to/vocab_classes.txt', "r")
    #     # for line in vocab_file:
    #     #     errorss = pr.test_gloss(line.split(" ")[1].strip(), e)
    #     #     print(errorss)
    #     # vocab_file.close()
    #
    #     # testing for specific glosses and running up MSE: errorss = pr.test_gloss("GLOSS", epoch, everyth sequence)
    #     # es = []
    #     # errorss = pr.test_gloss("ERZÄHLEN", e, 200)
    #     # es.append(errorss)
    #     # errorss = pr.test_gloss("ABEND", e, 200)
    #     # es.append(errorss)
    #     # ess = np.mean(np.asarray(es))
    #     # np.savez(args.out_test + 'e_fb' + str(e + 1) + 'error.npz', errors_mse=ess)
    #
    #     # testing for specific glosses feeding ground truth as next time step, and running up MSE:
    #     # es_GT = []
    #     # errorss = pr.sample_gloss_GT("ERZÄHLEN", e, 200)
    #     # es_GT.append(errorss)
    #     # errorss = pr.sample_gloss_GT("ABEND", e, 200)
    #     # es_GT.append(errorss)
    #     # ess_GT = np.mean(np.asarray(es_GT))
    #     # np.savez(args.out_test + 'e' + str(e + 1) + 'error.npz', errors_mse=ess, errors_GT_mse=ess_GT)



