# coding=utf-8
import numpy as np
import tensorflow as tf
from Ops import resblock_ll
import Utils as utils
import os.path
import pickle
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from sub_unit_predictor import subunit_predictor
from expert_filters import expert_filters

from make_p2v_maps import make_heatmaps


class pose_regressor(object):
    def __init__(self, args):
        # subunit network
        self.hidden_size = args.hidden_size
        self.num_coeff = args.num_coeff
        self.one_hot_size = args.one_hot_size

        # data
        self.X, self.Y, self.Xdim, self.Ydim, self.samples, self.Xmean, self.Xstd, self.Ymean, self.Ystd, self.lab, self.label_dim, self.Lmean, self.Lstd, self.labo, self.label_o_dim, self.Lomean, self.Lostd, self.L_den = self.get_data(
            args.data)
        self.X_nn = tf.placeholder(tf.float32, [None, self.Xdim], name='x-input')
        self.Y_nn = tf.placeholder(tf.float32, [None, self.Ydim], name='y-input')
        self.labels = tf.placeholder(tf.float32, [None, self.label_dim], name='label-input')
        self.labels_out = tf.placeholder(tf.float32, [None, self.label_o_dim], name='label-output')
        self.vocab_path = args.path_to_vocab

        # house-keeping
        self.batch_size = 1  # we do batch accumulation, set mini_batch_size to what you want your batch size to be
        self.mini_batch_size = args.mini_batch_size
        self.training_epochs = args.training_epochs

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prop')
        self.training = tf.placeholder(tf.bool, name='training')

        self.test_batch_size = 1
        self.inline_test = args.inline_test

        self.train = args.train

        self.out_model = args.out_model
        self.out_test = args.out_test
        self.out_log = args.out_log

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
            with open(self.out_model + 'inds.pkl', 'w') as f:
                pickle.dump([self.I_train, self.I_test], f)
        # batch size and epoch
        self.total_batch = int(self.samples / self.batch_size)
        print("total number of samples:", self.total_batch)
        # training set and  test set
        self.num_testBatch = np.int(self.I_test.size / self.batch_size)
        self.num_trainBatch = np.int(self.I_train.size / self.batch_size)
        print("training samples:", self.num_trainBatch)
        print("test samples:", self.num_testBatch)

        # optimizer
        self.learning_rate = args.learning_rate
        self.lr_c = tf.placeholder(tf.float32, name='lr_c')

        # expert filters
        self.filt0 = expert_filters([self.num_coeff, 2, self.Xdim / 2, 32], "filt0", self.rng)
        self.filt1 = expert_filters([self.num_coeff, 2, 32, self.Xdim / 2], "filt1", self.rng)

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

        # np.save("/vol/vssp/smile/tmp/nXmean.npy", Xmean)
        # np.save("/vol/vssp/smile/tmp/nXstd.npy", Xstd)
        # np.save("/vol/vssp/smile/tmp/nYmean.npy", Ymean)
        # np.save("/vol/vssp/smile/tmp/nYstd.npy", Ystd)
        # np.save("/vol/vssp/smile/tmp/nLmean.npy", Lmean)
        # np.save("/vol/vssp/smile/tmp/nLstd.npy", Lstd)
        # np.save("/vol/vssp/smile/tmp/nLomean.npy", Lomean)
        # np.save("/vol/vssp/smile/tmp/nLostd.npy", Lostd)

        return X, Y, Xdim, Ydim, samples, Xmean, Xstd, Ymean, Ystd, L, label_dim, Lmean, Lstd, Lo, label_o_dim, Lomean, Lostd, L_d

    def build_residual_model(self):
        class_input = self.labels
        x = tf.expand_dims(self.X_nn, -1)
        x = tf.reshape(x, [self.batch_size, 2, self.Xdim / 2])

        # pose classifier
        self.pose_class = subunit_predictor(class_input, self.label_dim, self.num_coeff, self.hidden_size, self.rng,
                                            self.keep_prob)

        self.classed = self.pose_class.classed

        self.bc_label = tf.concat([tf.transpose(self.classed), class_input[:, :self.one_hot_size]], 1)

        self.down_p = tf.contrib.layers.fully_connected(self.bc_label, self.num_coeff, activation_fn=None)

        # pose regressor
        self.H0 = resblock_ll(x, self.filt0.get_filt(tf.transpose(self.down_p), self.batch_size), name='layer0')
        self.H1 = resblock_ll(self.H0, self.filt1.get_filt(tf.transpose(self.down_p), self.batch_size), name='layer1')

        self.H4 = tf.layers.flatten(self.H1)
        self.F3 = tf.contrib.layers.fully_connected(self.H4, self.Ydim, activation_fn=None)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.square(self.Y_nn - tf.squeeze(self.F3)))
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("optimizer"):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.lr_c)

            tvs = tf.trainable_variables()
            # Creation of a list of variables with the same shape as the trainable ones, initialized with 0s
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
            self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

            # Calls the compute_gradients function of the optimizer to obtain... the list of gradients
            gvs = self.optim.compute_gradients(self.loss, tvs)

            # Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
            self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

            # Define the training step (part with variable value update)
            self.train_step = self.optim.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    def train_nn(self):
        # session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(self.out_log, sess.graph)

        # saver for saving the variables
        saver = tf.train.Saver(max_to_keep=10000)
        ii = 0
        try:
            check_dir = self.out_model
            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            epoch_str = chkpt_fname.split("-")[-1]
            saver.restore(sess, chkpt_fname)
            ii = self.num_trainBatch * int(epoch_str)
        except:
            print("No model found, starting training from scratch!")
            epoch_str = "0"

        for epoch in range(int(epoch_str) + 1, self.training_epochs + 1):
            avg_cost_train = 0
            b = 0
            for i in range(self.num_trainBatch):
                if b == 0:
                    sess.run(self.zero_ops)
                if b < self.mini_batch_size:
                    index_train = self.I_train[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_xs = self.X[index_train]
                    batch_ys = self.Y[index_train]
                    batch_labels = self.lab[index_train]
                    # batch_labels_out = self.labo[index_train]

                    feed_dict = {self.X_nn: batch_xs, self.Y_nn: batch_ys, self.labels: batch_labels,
                                 self.keep_prob: 0.7, self.lr_c: self.learning_rate,
                                 self.training: True}
                    classed, down_p, sample, l, _ = sess.run(
                        [self.classed, self.down_p, self.F3, self.loss, self.accum_ops], feed_dict=feed_dict)
                    avg_cost_train += l / self.num_trainBatch
                    if index_train < (len(self.Y) - 1):
                        batch_ys = self.Y[index_train + 1]
                        batch_labels = np.hstack((batch_labels[:, :self.one_hot_size], sample))
                        feed_dict = {self.X_nn: sample, self.Y_nn: batch_ys, self.labels: batch_labels,
                                     self.keep_prob: 0.7, self.lr_c: self.learning_rate,
                                     self.training: True}

                        classed, down_p, l, _ = sess.run([self.classed, self.down_p, self.loss, self.accum_ops],
                                                         feed_dict=feed_dict)
                        avg_cost_train += l / self.num_trainBatch
                    b += 1

                else:
                    l, _ = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
                    avg_cost_train += l / self.num_trainBatch
                    b = 0
                if i % 1000 == 0:
                    avg_cost_train += l / self.num_trainBatch
                    print('Epoch: %04d' % epoch, 'total step: %04d' % (i + ii), 'trainingloss = {:.9f}'.format(l))
                    merged = tf.summary.merge_all()
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, i + ii)

            # save model and weights
            saver.save(sess, self.out_model + "out", global_step=epoch)
            print('Epoch: %04d' % epoch, 'trainingloss = {:.9f}'.format(avg_cost_train))

            # test
            # if(self.inline_test):
            # self.test_gloss_inline("ABER", epoch, sess)

        sess.close()

    def test_sequence_inline(self, epoch, sess):
        out = self.out_test + 'figs_seq/' + str(epoch) + '/'
        if not os.path.exists(out):
            os.makedirs(out)
        class_file = open(out + 'classes.txt', "w")
        for i in range(self.num_testBatch):
            if i % 1000 == 0:
                index_test = self.I_test[i * self.test_batch_size: (i + 1) * self.test_batch_size]
                batch_xs = self.X[index_test]
                batch_labels = self.lab[index_test]
                batch_labels_out = self.labo[index_test]
                batch_labels_d = np.squeeze(batch_labels_out)
                batch_labels_d = batch_labels_d[:self.one_hot_size] * self.Lostd[:self.one_hot_size] + self.Lomean[
                                                                                                       :self.one_hot_size]

                out = self.out_test + 'figs_seq/' + str(epoch) + '/batch' + str(i) + '/'
                if not os.path.exists(out):
                    os.makedirs(out)

                samples_seq = []
                feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
                classed, down_p, samples_0 = sess.run([self.classed, self.down_p, self.F3], feed_dict=feed_dict)
                class_file.write(str(i) + " gt: " + str(batch_labels_d[:]) + ", predicted: " + str(
                    classed[:, 0]) + ", coefficients: " + str(down_p[0, :]) + "\n")

                samples_0d = samples_0 * self.Ystd + self.Ymean
                samples_seq.append(samples_0d)
                samples = samples_0
                for k in range(150):
                    feed_dict = {self.X_nn: samples, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
                    classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)
                    samples_d = samples * self.Ystd + self.Ymean
                    samples_seq.append(samples_d)

                for j in range(0, self.batch_size):
                    for k in range(len(samples_seq)):
                        sample = samples_seq[k]
                        x = sample[j, 0:252:2]  # predicted samples
                        y = sample[j, 1:252:2]  # predicted samples

                        plt.plot(x, y, 'ro')
                        plt.gca().set_xlim([700, 1300])
                        plt.gca().set_ylim([100, 900])
                        plt.gca().invert_yaxis()
                        plt.savefig(out + str(j) + '_' + str(k) + '.png')
                        plt.close()
        class_file.close()

    def test_sequence(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        check_dir = self.out_model
        chkpt_fname = tf.train.latest_checkpoint(check_dir)
        epoch = chkpt_fname.split("-")[-1]
        saver.restore(sess, chkpt_fname)
        out = self.out_test + 'figs_seq_by_vocab_e' + epoch.encode("utf8") + '/'
        if not os.path.exists(out):
            os.makedirs(out)
        class_file = open(out + 'classes.txt', "w")
        vocab_file = open('./vocab_classes.txt', "r")

        for i in range(self.num_testBatch):
            if i % 500 == 0:
                index_test = self.I_test[i * self.test_batch_size: (i + 1) * self.test_batch_size]
                batch_xs = self.X[index_test]
                batch_labels = self.lab[index_test]
                batch_labels_out = self.labo[index_test]
                batch_labels_d = np.squeeze(batch_labels)
                batch_labels_d = batch_labels_d[:self.one_hot_size] * self.Lstd[:self.one_hot_size] + self.Lmean[
                                                                                                      :self.one_hot_size]

                batch_labels_d_out = np.squeeze(batch_labels_out)
                batch_labels_d_out = batch_labels_d_out[:self.one_hot_size] * self.Lostd[
                                                                              :self.one_hot_size] + self.Lomean[
                                                                                                    :self.one_hot_size]

                voc = ""
                vi = [o for o, x in enumerate(batch_labels_d) if x > 0.0]
                vocab_file.seek(0)
                for j, line in enumerate(vocab_file):
                    if j == vi[0]:
                        voc = line.strip().split()[-1]

                        break

                out = self.out_test + 'figs_seq_by_vocab_e' + epoch.encode("utf8") + '/' + voc + '_' + str(i) + '/'
                if not os.path.exists(out):
                    os.makedirs(out)

                samples_seq = []
                feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
                classed, down_p, samples_0 = sess.run([self.classed, self.down_p, self.F3], feed_dict=feed_dict)

                print('class: ' + str(classed[:, 0]))

                samples_0d = samples_0 * self.Ystd + self.Ymean
                samples_seq.append(samples_0d)
                samples = samples_0
                # samples_tm1 = samples_0
                for k in range(150):
                    feed_dict = {self.X_nn: samples, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
                    classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)
                    samples_d = samples * self.Ystd + self.Ymean
                    samples_seq.append(samples_d)

                for j in range(0, self.batch_size):
                    for k in range(len(samples_seq)):
                        sample = samples_seq[k]
                        x = sample[j, 0:252:2]  # predicted samples
                        y = sample[j, 1:252:2]  # predicted samples

                        plt.plot(x, y, 'ro')
                        plt.gca().set_xlim([700, 1300])
                        plt.gca().set_ylim([100, 900])
                        plt.gca().invert_yaxis()
                        plt.savefig(out + str(j) + '_' + str(k) + '.png')
                        plt.close()
        class_file.close()
        vocab_file.close()
        sess.close()
        return

    def test_gloss(self, gloss, epoch, everyth):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        check_dir = self.out_model
        chkpt_fname = tf.train.latest_checkpoint(check_dir)
        chksplit = chkpt_fname.split("-")
        chksplit[-1] = str(epoch)
        chkpt_fname = "-".join(chksplit)
        print(chkpt_fname)
        try:
            saver.restore(sess, chkpt_fname)
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
                    batch_ys = np.expand_dims(batch_ys, 0)
                    batch_labels = np.expand_dims(batch_labels, 0)
                    feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                 self.training: False}
                    classed, down_p, samples_0 = sess.run([self.classed, self.down_p, self.F3], feed_dict=feed_dict)

                    samples_0d = samples_0 * self.Ystd + self.Ymean
                    samples_seq.append(samples_0d)
                    samples = samples_0
                    err_frame = batch_ys[0, :252] - samples[0, :252]
                    errors_per_frame.append(np.mean(np.square(np.squeeze(err_frame))))
                    for k in range(150):
                        feed_dict = {self.X_nn: samples, self.labels: batch_labels, self.keep_prob: 1,
                                     self.training: False}
                        classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                        samples_d = samples * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)
                        batch_labels = np.hstack((batch_labels[0, :self.one_hot_size], samples[0, :]))
                        batch_labels = np.expand_dims(batch_labels, 0)
                        err_frame = batch_ys[0, :252] - samples[0, :252]
                        errors_per_frame.append(np.mean(np.square(np.squeeze(err_frame))))

                    errors_frame_sum = np.mean(np.asarray(errors_per_frame))
                    errors_per_frame = []
                    errors_per_seq.append(errors_frame_sum)

                    X = []
                    Y = []

                    for j in range(0, self.batch_size):
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
        sess.close()
        return gloss_error

    def test_gloss_inline(self, gloss, epoch, sess):

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

                    out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss + '_e' + str(
                        epoch) + '/' + gloss + '_test' + str(x) + '/'
                    if not os.path.exists(out):
                        os.makedirs(out)

                    samples_seq = []
                    batch_xs = np.expand_dims(batch_xs, 0)
                    batch_labels = np.expand_dims(batch_labels, 0)
                    feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                 self.training: False}
                    classed, down_p, samples_0 = sess.run([self.classed, self.down_p, self.F3], feed_dict=feed_dict)

                    samples_0d = samples_0 * self.Ystd + self.Ymean
                    samples_seq.append(samples_0d)
                    samples = samples_0
                    for k in range(150):
                        feed_dict = {self.X_nn: samples, self.labels: batch_labels, self.keep_prob: 1,
                                     self.training: False}
                        classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                        samples_d = samples * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)

                    for j in range(0, self.batch_size):
                        for k in range(len(samples_seq)):
                            sample = samples_seq[k]
                            x = sample[j, 0:252:2]  # predicted samples
                            y = sample[j, 1:252:2]  # predicted samples

                            fp = out + str(j) + '_' + str(k).zfill(3) + '.png'
                            make_heatmaps(sample[j, :], fp)
                            plt.plot(x, y, 'ro')
                            plt.gca().set_xlim([700, 1300])
                            plt.gca().set_ylim([100, 900])
                            plt.gca().invert_yaxis()
                            plt.savefig(out + str(j) + '_' + str(k).zfill(3) + '.png')
                            plt.close()

        vocab_file.close()
        return

    def test_gloss_seq(self, glosses, epoch, gloss_length, everyth):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        check_dir = self.out_model
        chkpt_fname = tf.train.latest_checkpoint(check_dir)
        chksplit = chkpt_fname.split("-")
        chksplit[-1] = str(epoch)
        chkpt_fname = "-".join(chksplit)

        try:
            saver.restore(sess, chkpt_fname)
        except:
            print("No model found! Exiting.")
            return

        # epoch += 1
        gloss_list = glosses.split(" ")
        glosses = gloss_list
        gloss_str = "_".join(gloss_list)
        out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss_str + '_e' + str(epoch) + '/'
        if not os.path.exists(out):
            os.makedirs(out)
        vocab_file = open(self.vocab_path, "r")

        one_hots = []
        for gloss in glosses:
            one_hot_index = -1
            for line in vocab_file:
                if gloss in line:
                    one_hot_index = int(line.split()[0])
            if one_hot_index > -1:
                one_hot = np.zeros(self.one_hot_size)
                one_hot[one_hot_index] = 1.0
                one_hots.append(one_hot)
            vocab_file.seek(0)

        idd = [i for i in self.I_test if all(self.L_den[i, :] == one_hots[0])]

        for x in range(len(idd)):
            if x % everyth == 0:
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

                out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss_str + '_e' + str(
                    epoch) + '/' + gloss_str + '_test' + str(idd[x]) + '/'
                print(out)
                if not os.path.exists(out):
                    os.makedirs(out)

                samples_seq = []
                batch_xs = np.expand_dims(batch_xs, 0)
                batch_labels = np.expand_dims(batch_labels, 0)
                feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
                classed, down_p, samples_0 = sess.run([self.classed, self.down_p, self.F3], feed_dict=feed_dict)

                samples_0d = samples_0 * self.Ystd + self.Ymean
                samples_seq.append(samples_0d)
                samples = samples_0
                for k in range(gloss_length):
                    feed_dict = {self.X_nn: samples, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
                    classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                    samples_d = samples * self.Ystd + self.Ymean
                    samples_seq.append(samples_d)

                    fp = out + glosses[0] + '_' + str(k).zfill(3) + '.png'
                    make_heatmaps(samples_d[0, :], fp)
                    batch_labels = np.hstack((batch_labels[0, :self.one_hot_size], samples[0, :]))
                    batch_labels = np.expand_dims(batch_labels, 0)

                for h in range(1, len(one_hots)):
                    one_hot_one_n = (one_hots[h] - self.Lmean[:self.one_hot_size]) / self.Lstd[:self.one_hot_size]
                    batch_labels = np.hstack((one_hot_one_n, samples[0, :]))
                    batch_labels = np.expand_dims(batch_labels, 0)
                    for k in range(gloss_length):
                        feed_dict = {self.X_nn: samples, self.labels: batch_labels, self.keep_prob: 1,
                                     self.training: False}
                        classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                        samples_d = samples * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)

                        fp = out + glosses[h] + '_' + str(k).zfill(3) + '.png'
                        make_heatmaps(samples_d[0, :], fp)
                        batch_labels = np.hstack((one_hot_one_n, samples[0, :]))
                        batch_labels = np.expand_dims(batch_labels, 0)

        vocab_file.close()
        sess.close()
        return

    def sample_gloss_GT(self, gloss, epoch, everyth):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        check_dir = self.out_model
        chkpt_fname = tf.train.latest_checkpoint(check_dir)
        chksplit = chkpt_fname.split("-")
        chksplit[-1] = str(epoch)
        chkpt_fname = "-".join(chksplit)
        print(chkpt_fname)
        try:
            saver.restore(sess, chkpt_fname)
        except:
            print("No model found! Exiting.")
            return

        # epoch += 1
        out = self.out_test + 'e' + str(epoch) + '_GTmaps/' + gloss + '_e' + str(epoch) + '/'
        print(out)
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
                    batch_labels_d = self.L_den[idd[x]]

                    out = self.out_test + 'e' + str(epoch) + '_GTmaps/' + gloss + '_e' + str(
                        epoch) + '/' + gloss + '_test' + str(idd[x]) + '/'
                    print(out)
                    if not os.path.exists(out):
                        os.makedirs(out)

                    out_pose = self.out_test + 'e' + str(epoch) + '_GTpose/' + gloss + '_e' + str(
                        epoch) + '/' + gloss + '_test' + str(idd[x]) + '/'
                    print(out_pose)

                    if not os.path.exists(out_pose):
                        os.makedirs(out_pose)

                    samples_seq = []
                    batch_xs = np.expand_dims(batch_xs, 0)
                    batch_labels = np.expand_dims(batch_labels, 0)
                    feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                 self.training: False}
                    classed, samples_0 = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                    samples_0d = samples_0 * self.Ystd + self.Ymean
                    samples_seq.append(samples_0d)
                    samples = samples_0
                    err_frame = batch_ys[:252] - samples[0, :252]
                    errors_per_frame.append(np.mean(np.square(np.squeeze(err_frame))))
                    u = 0
                    while all(batch_labels_d == one_hot) and u <= 150:
                        u += 1
                        batch_xs = self.X[idd[x] + u]
                        batch_ys = self.Y[idd[x]]
                        batch_labels = self.lab[idd[x] + u]
                        batch_labels_d = self.L_den[idd[x] + u]

                        batch_xs = np.expand_dims(batch_xs, 0)
                        batch_labels = np.expand_dims(batch_labels, 0)

                        feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                     self.training: False}
                        classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                        samples_d = samples * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)
                        err_frame = batch_ys[:252] - samples[0, :252]
                        errors_per_frame.append(np.mean(np.square(np.squeeze(err_frame))))

                    X = []
                    Y = []

                    for j in range(0, self.batch_size):
                        for k in range(len(samples_seq)):
                            sample = samples_seq[k]
                            X.append(sample[j, 0:504:2])
                            Y.append(sample[j, 1:504:2])
                            x = sample[j, 0:252:2]  # predicted samples
                            y = sample[j, 1:252:2]  # predicted samples

                            # fp = out + str(j) + '_' + str(k).zfill(3) + '.png'
                            # make_heatmaps(sample[j, :], fp)
                            # plt.plot(x, y, 'ro')
                            # plt.gca().set_xlim([700, 1300])
                            # plt.gca().set_ylim([100, 900])
                            # plt.gca().invert_yaxis()
                            # plt.savefig(out_pose + str(j) + '_' + str(k).zfill(3) + '.png')
                            # plt.close()
                    np.save(out_pose + '/Xsamples.npy', np.asarray(X))
                    np.save(out_pose + '/Ysamples.npy', np.asarray(Y))
                    errors_frame_sum = np.mean(np.asarray(errors_per_frame))
                    errors_per_frame = []
                    errors_per_seq.append(errors_frame_sum)

        gloss_error = np.mean(np.asarray(errors_per_seq))
        np.savez(self.out_test + 'e' + str(epoch) + 'maps/' + gloss + '_e' + str(epoch) + '/error.npz',
                 errors_mse=gloss_error)
        vocab_file.close()
        sess.close()
        return gloss_error

    def sample_gloss_GT_inline(self, gloss, epoch, sess):

        out = self.out_test + 'e' + str(epoch) + '_GTmaps/' + gloss + '_e' + str(epoch) + '/'
        print(out)
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
                    batch_labels_d = self.L_den[idd[x]]

                    out = self.out_test + 'e' + str(epoch) + '_GTmaps/' + gloss + '_e' + str(
                        epoch) + '/' + gloss + '_' + str(x) + '/'
                    print(out)
                    if not os.path.exists(out):
                        os.makedirs(out)

                    samples_seq = []
                    batch_xs = np.expand_dims(batch_xs, 0)
                    batch_labels = np.expand_dims(batch_labels, 0)
                    feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                 self.training: False}
                    classed, samples_0 = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                    samples_0d = samples_0 * self.Ystd + self.Ymean
                    samples_seq.append(samples_0d)
                    u = 0
                    while all(batch_labels_d == one_hot) and u <= 150:
                        u += 1
                        batch_xs = self.X[idd[x] + u]
                        batch_labels = self.lab[idd[x] + u]
                        batch_labels_d = self.L_den[idd[x] + u]

                        batch_xs = np.expand_dims(batch_xs, 0)
                        batch_labels = np.expand_dims(batch_labels, 0)

                        feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                     self.training: False}
                        classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                        samples_d = samples * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)

                    for j in range(0, self.batch_size):
                        for k in range(len(samples_seq)):
                            sample = samples_seq[k]
                            x = sample[j, 0:252:2]  # predicted samples
                            y = sample[j, 1:252:2]  # predicted samples

                            fp = out + str(j) + '_' + str(k).zfill(3) + '.png'
                            make_heatmaps(sample[j, :], fp)
                            plt.plot(x, y, 'ro')
                            plt.gca().set_xlim([700, 1300])
                            plt.gca().set_ylim([100, 900])
                            plt.gca().invert_yaxis()
                            plt.savefig(out + str(j) + '_' + str(k).zfill(3) + '.png')
                            plt.close()

        vocab_file.close()
        return

    def test_gloss_seq_GT(self, glosses, epoch, everyth):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        check_dir = self.out_model
        chkpt_fname = tf.train.latest_checkpoint(check_dir)
        chksplit = chkpt_fname.split("-")
        chksplit[-1] = str(epoch)
        chkpt_fname = "-".join(chksplit)

        try:
            saver.restore(sess, chkpt_fname)
        except:
            print("No model found! Exiting.")
            return

        gloss_list = glosses.split(" ")
        glosses = gloss_list
        gloss_str = "_".join(gloss_list)
        out = self.out_test + 'e' + str(epoch) + 'maps/' + gloss_str + '_e' + str(epoch) + '/'
        if not os.path.exists(out):
            os.makedirs(out)
        vocab_file = open(self.vocab_path, "r")

        one_hots = []
        for gloss in glosses:
            one_hot_index = -1
            for line in vocab_file:
                if gloss in line:
                    one_hot_index = int(line.split()[0])

            if one_hot_index > -1:
                one_hot = np.zeros(self.one_hot_size)
                one_hot[one_hot_index] = 1.0
                one_hots.append(one_hot)
            vocab_file.seek(0)

        for one_hot, gloss in zip(one_hots, glosses):

            idd = [i for i in self.I_test if all(self.L_den[i, :] == one_hots[0])]

            for x in range(len(idd)):
                if x % everyth == 0:
                    batch_xs = self.X[idd[x]]
                    batch_labels = self.lab[idd[x]]
                    batch_labels_d = self.L_den[idd[x]]

                    out = self.out_test + 'e' + str(epoch) + '_GTmaps/' + gloss + '_e' + str(
                        epoch) + '/' + gloss + '_' + str(x) + '/'
                    print(out)
                    if not os.path.exists(out):
                        os.makedirs(out)

                    samples_seq = []
                    batch_xs = np.expand_dims(batch_xs, 0)
                    batch_labels = np.expand_dims(batch_labels, 0)
                    feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                 self.training: False}
                    classed, samples_0 = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                    samples_0d = samples_0 * self.Ystd + self.Ymean
                    samples_seq.append(samples_0d)
                    u = 0
                    while all(batch_labels_d == one_hot) and u <= 45:
                        u += 1
                        batch_xs = self.X[idd[x] + u]
                        batch_labels = self.lab[idd[x] + u]
                        batch_labels_d = self.L_den[idd[x] + u]

                        batch_xs = np.expand_dims(batch_xs, 0)
                        batch_labels = np.expand_dims(batch_labels, 0)

                        feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1,
                                     self.training: False}
                        classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                        samples_d = samples * self.Ystd + self.Ymean
                        samples_seq.append(samples_d)

                        fp = out + gloss + '_' + str(u).zfill(3) + '.png'
                        make_heatmaps(samples_d[0, :], fp)

        vocab_file.close()
        sess.close()
        return

    def gen_from_test_inds(self, _inds_npz, gloss, epoch):
        data = np.load(_inds_npz)
        inds = data['inds']

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        check_dir = self.out_model
        chkpt_fname = tf.train.latest_checkpoint(check_dir)
        chksplit = chkpt_fname.split("-")
        chksplit[-1] = str(epoch)
        chkpt_fname = "-".join(chksplit)

        try:
            saver.restore(sess, chkpt_fname)
        except:
            print("No model found! Exiting.")
            return

        epoch += 1

        for x in range(len(inds)):
            batch_xs = self.X[inds[x]]
            batch_ys = self.Y[inds[x]]
            batch_labels = self.lab[inds[x]]
            batch_labels_out = self.labo[inds[x]]
            batch_labels_d = np.squeeze(batch_labels)
            batch_labels_d = batch_labels_d[:self.one_hot_size] * self.Lstd[:self.one_hot_size] + self.Lmean[
                                                                                                  :self.one_hot_size]

            batch_labels_d_out = np.squeeze(batch_labels_out)
            batch_labels_d_out = batch_labels_d_out[:self.one_hot_size] * self.Lostd[
                                                                          :self.one_hot_size] + self.Lomean[
                                                                                                :self.one_hot_size]

            out = self.out_test + 'e' + str(epoch) + 'maps_for_g2p2v/' + gloss + '_e' + str(
                epoch) + '/' + gloss + '_test' + str(
                inds[x]) + '/'
            print(out)
            if not os.path.exists(out):
                os.makedirs(out)

            out_pose = self.out_test + 'e' + str(epoch) + 'pose_for_g2p2v/' + gloss + '_e' + str(
                epoch) + '/' + gloss + '_test' + str(inds[x]) + '/'
            print(out_pose)

            if not os.path.exists(out_pose):
                os.makedirs(out_pose)

            samples_seq = []
            batch_xs = np.expand_dims(batch_xs, 0)
            batch_ys = np.expand_dims(batch_ys, 0)
            batch_labels = np.expand_dims(batch_labels, 0)
            feed_dict = {self.X_nn: batch_xs, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
            classed, down_p, samples_0 = sess.run([self.classed, self.down_p, self.F3], feed_dict=feed_dict)

            samples_0d = samples_0 * self.Ystd + self.Ymean
            samples_seq.append(samples_0d)
            samples = samples_0

            for k in range(150):
                feed_dict = {self.X_nn: samples, self.labels: batch_labels, self.keep_prob: 1, self.training: False}
                classed, samples = sess.run([self.classed, self.F3], feed_dict=feed_dict)

                samples_d = samples * self.Ystd + self.Ymean
                samples_seq.append(samples_d)
                batch_labels = np.hstack((batch_labels[0, :self.one_hot_size], samples[0, :]))
                batch_labels = np.expand_dims(batch_labels, 0)

            for j in range(0, self.batch_size):
                for k in range(len(samples_seq)):
                    sample = samples_seq[k]
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

        sess.close()
        return


# tf.set_random_seed(23456)

args = utils.parse_args()
utils.build_path([args.out_model])
utils.build_path([args.out_test])
utils.build_path([args.out_log])

pr = pose_regressor(args)
pr.build_residual_model()

if args.train:
    pr.train_nn()
else:

    # for testing sequences across whole test set from last epoch: pr.test_sequence()

    # for specific epoch(s) and glosses:
    for e in range(5, 6):
        # testing for specific glosses using all instances in the test dataset:
        # inds_npz = "/path/to/gloss/inds/GLOSS_inds.npz"
        # pr.gen_from_test_inds(inds_npz, "GLOSS", e)

        # testing for specific glosses and running up MSE: errorss = pr.test_gloss("GLOSS", epoch, everyth sequence)
        es = []
        errorss = pr.test_gloss("ERZÄHLEN", e, 200)
        es.append(errorss)
        errorss = pr.test_gloss("ABEND", e, 200)
        es.append(errorss)
        errorss = pr.test_gloss("ABER", e, 200)
        es.append(errorss)
        errorss = pr.test_gloss("JAHR", e, 200)
        es.append(errorss)
        errorss = pr.test_gloss("VORGESTERN", e, 200)
        es.append(errorss)
        ess = np.mean(np.asarray(es))
        np.savez(args.out_test + 'e_fb' + str(e + 1) + 'error.npz', errors_mse=ess)

        # testing for specific glosses feeding ground truth as next time step, and running up MSE:
        es_GT = []
        errorss = pr.sample_gloss_GT("ERZÄHLEN", e, 200)
        es_GT.append(errorss)
        errorss = pr.sample_gloss_GT("ABEND", e, 200)
        es_GT.append(errorss)
        errorss = pr.sample_gloss_GT("ABER", e, 200)
        es_GT.append(errorss)
        errorss = pr.sample_gloss_GT("JAHR", e, 200)
        es_GT.append(errorss)
        errorss = pr.sample_gloss_GT("VORGESTERN", e, 200)
        es_GT.append(errorss)
        ess_GT = np.mean(np.asarray(es_GT))
        np.savez(args.out_test + 'e' + str(e + 1) + 'error.npz', errors_mse=ess, errors_GT_mse=ess_GT)

        # testing for gloss sequences: pr.test_gloss_seq("GLOSS SEQUENCE", epoch, gloss_length, everyth sequence)
        pr.test_gloss_seq("JETZT SPIELEN", 20, 60, 100)
        pr.test_gloss_seq("JETZT FUSSBALL SPIELEN", 20, 60, 100)

        # for GT fed sequences use: pr.test_gloss_seq_GT(glosses, epoch, everyth)

        # testing for whole vocabulary:
        # vocab_file = open('/path/to/vocab_classes.txt', "r")
        # for line in vocab_file:
        #     errorss = pr.test_gloss(line.split(" ")[1].strip(), e)
        #     print(errorss)
        # vocab_file.close()
