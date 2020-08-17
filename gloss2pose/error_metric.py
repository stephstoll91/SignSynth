# coding=utf-8
import numpy as np
import math

import scipy
import scipy.io as sio
#from scipy.spatial import procrustes
from scipy.signal import find_peaks
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from clustering import cluster_and_plot


def get_baseline_skels(gt_dir, glosses):

    all_GT = []
    all_gloss_gt = []
    all_probs = []

    for gt_file in os.listdir(gt_dir):
        if any(_gloss in gt_file for _gloss in glosses):
            gt_data = sio.loadmat(gt_dir + gt_file)
            gt_input = gt_data['input']
            GT_gloss = str.split(gt_file, '_')[-2]

            for l in range(6, 7):
                GT = []
                #normed_input_GT = normalise(gt_input, l)
                normed_input_GT = gt_input
                pose_GT = normed_input_GT[0, l]['pose']
                face_GT = normed_input_GT[0, l]['face']
                hl_GT = normed_input_GT[0, l]['hand_l']
                hr_GT = normed_input_GT[0, l]['hand_r']

                for h in range(0, len(pose_GT)):
                    pose_GT[h] = scale(pose_GT[h], 0, 10, True)
                    face_GT[h] = scale(face_GT[h], 0, 10, True)
                    hl_GT[h] = scale(hl_GT[h], 0, 10, True)
                    hr_GT[h] = scale(hr_GT[h], 0, 10, True)


                    GT_line = np.hstack((pose_GT[h], face_GT[h],
                                         hl_GT[h], hr_GT[h]))

                    GT.append(GT_line)

                GT = np.asarray(GT)

                Xstd = GT.std(axis=0)
                for i in range(Xstd.size):
                    if (Xstd[i] == 0):
                        Xstd[i] = 1

                GT = (GT - GT.mean(axis=0)) / Xstd
                all_GT.append(np.asarray(GT))
                all_gloss_gt.append(GT_gloss)

    return all_GT, all_gloss_gt, all_probs


def apply_procrustes(ref, now, l):

    transformed = []
    now_frame = np.reshape(now[0], (-1, 2))
    disp, transformed_points, rot_scale_trans = procrustes(ref, now_frame)

    for p in range(l):
        frame = np.reshape(now[p], (-1, 2))

        # c — Translation component
        # T — Orthogonal rotation and reflection component
        # b — Scale component
        # Z = b * Y * T + c;

        frame_trans = np.dot(rot_scale_trans['scale'], frame)

        frame_trans = np.dot(frame_trans, rot_scale_trans['rotation'])

        frame_trans = frame_trans + rot_scale_trans['translation']

        # plt.figure()
        # plt.scatter(ref[:, 0], ref[:, 1])
        # plt.scatter(transformed_points[:, 0], transformed_points[:, 1])
        # plt.scatter(frame_trans[:, 0], frame_trans[:, 1])
        # plt.gca().invert_yaxis()
        # plt.savefig('./norms/' + str(p) + '.png')
        # plt.close()

        transformed.append(frame_trans.flatten())

    return np.asarray(transformed), rot_scale_trans


def procrustes(X, Y, scaling=False, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1
python -m pip install --upgrade pip
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def normalise(_input, _l):
    #load reference skeleton
    ref_pose = np.load('/home/steph/Documents/Chapter2/Code/smile-tmp/Mollica-ref-pose.npy')[0]
    #ref_face = np.load('Mollica-ref-face.npy')[0]
    #ref_hl = np.load('Mollica-ref-hand-left.npy')[0]
    #ref_hr = np.load('Mollica-ref-hand-right.npy')[0]

    ref_pose = np.reshape(ref_pose, (-1, 2))
    #ref_face = np.reshape(ref_face, (-1, 2))
    #ref_hl = np.reshape(ref_hl, (-1, 2))
    #ref_hr = np.reshape(ref_hr, (-1, 2))

    pose = _input[0, _l]['pose']
    face = _input[0,_l]['face']
    hl = _input[0,_l]['hand_l']
    hr = _input[0, _l]['hand_r']

    poseT, rst = apply_procrustes(ref_pose, pose, len(pose))

    faceT = []
    hlT = []
    hrT = []
    for p in range(len(pose)):
        frame = np.reshape(face[p], (-1, 2))
        frame_trans = np.dot(rst['scale'], frame)
        frame_trans = np.dot(frame_trans, rst['rotation'])
        frame_trans = frame_trans + rst['translation']

        faceT.append(frame_trans.flatten())

        frame = np.reshape(hl[p], (-1, 2))
        frame_trans = np.dot(rst['scale'], frame)
        frame_trans = np.dot(frame_trans, rst['rotation'])
        frame_trans = frame_trans + rst['translation']

        hlT.append(frame_trans.flatten())

        frame = np.reshape(hr[p], (-1, 2))
        frame_trans = np.dot(rst['scale'], frame)
        frame_trans = np.dot(frame_trans, rst['rotation'])
        frame_trans = frame_trans + rst['translation']

        hrT.append(frame_trans.flatten())



    # faceT = apply_procrustes(ref_face, face, len(face))
    # hlT = apply_procrustes(ref_hl, hl, len(hl))
    # hrT = apply_procrustes(ref_hr, hr, len(hr))


    _input[0, _l]['pose'] = poseT
    _input[0, _l]['face'] = np.asarray(faceT)
    _input[0, _l]['hand_l'] = np.asarray(hlT)
    _input[0, _l]['hand_r'] = np.asarray(hrT)

    return _input

def scale(data, a, b, flip):

    frame = np.reshape(data, (-1, 2))

    if flip:

        #frame[:, 0] = ((b - a) * (frame[:, 0] - np.min(frame[:, 0])) / (
                #np.max(frame[:, 0]) - np.min(frame[:, 0]) + 0.000000001)) + a

        frame[:, 1] = -1 * (frame[:, 1] - np.min(frame[:, 1])) / (
               np.max(frame[:, 1]) - np.min(frame[:, 1]) + 0.000000001) + 1

        #frame[:, 1] = ((b - a) * (frame[:, 1] - np.min(frame[:, 1])) / (
                #np.max(frame[:, 1]) - np.min(frame[:, 1]) + 0.000000001)) + a

    #else:
        #frame[:, 0] = ((b - a) * (frame[:, 0] - np.min(frame[:, 0])) / (
                #np.max(frame[:, 0]) - np.min(frame[:, 0]) + 0.000000001)) + a
        #frame[:, 1] = ((b - a) * (frame[:, 1] - np.min(frame[:, 1])) / (
                #np.max(frame[:, 1]) - np.min(frame[:, 1]) + 0.000000001)) + a

    else:
        plt.figure()
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.show()
        plt.close()

    data = np.reshape(frame, (-1))
    return data

def collect_data(test_dir, gt_dir, _glosses):

    all_test = []
    all_GT = []
    all_gloss_test= []
    all_gloss_gt = []
    all_probs_test = []
    GT = []
    test = []
    probs_test = []



    # _gloss = test_file.split('_')[0]
    for gt_file in os.listdir(gt_dir):
        if any(_gloss in gt_file for _gloss in _glosses):
            gt_data = sio.loadmat(gt_dir + gt_file)
            gt_input = gt_data['input']
            GT_gloss = str.split(gt_file, '_')[-2]

            for l in range(0, 6):
                GT = []
                normed_input_GT = normalise(gt_input, l)
                #normed_input_GT = gt_input
                pose_GT = normed_input_GT[0, l]['pose']
                face_GT = normed_input_GT[0, l]['face']
                hl_GT = normed_input_GT[0, l]['hand_l']
                hr_GT = normed_input_GT[0, l]['hand_r']

                for h in range(0, len(pose_GT)):
                    pose_GT[h] = scale(pose_GT[h], 0, 10, True)
                    face_GT[h] = scale(face_GT[h], 0, 10, True)
                    hl_GT[h] = scale(hl_GT[h], 0, 10, True)
                    hr_GT[h] = scale(hr_GT[h], 0, 10, True)

                    GT_line = np.hstack((pose_GT[h], face_GT[h],
                                         hl_GT[h], hr_GT[h]))

                    GT.append(GT_line)

                GT = np.asarray(GT)

                Xstd = GT.std(axis=0)
                for i in range(Xstd.size):
                    if (Xstd[i] == 0):
                        Xstd[i] = 1

                GT = (GT - GT.mean(axis=0)) / Xstd
                all_GT.append(np.asarray(GT))
                all_gloss_gt.append(GT_gloss)

    #all_test, all_gloss_test, all_probs_test = get_baseline_skels(gt_dir, _glosses)

    for test_file in os.listdir(test_dir):
        if os.path.isfile(test_dir + test_file):
            test_data = sio.loadmat(test_dir + test_file)
            test_input = test_data['input']
            test_gloss = str.split(test_file, '_')[0]

            normed_input_test = test_input
            pose_test = normed_input_test[0, 0]['pose']
            face_test = normed_input_test[0, 0]['face']
            hl_test = normed_input_test[0, 0]['hand_l']
            hr_test = normed_input_test[0, 0]['hand_r']

            pose_test_prob = normed_input_test[0, 0]['prob_pose']
            face_test_prob = normed_input_test[0, 0]['prob_face']
            hl_test_prob = normed_input_test[0, 0]['prob_hand_l']
            hr_test_prob = normed_input_test[0, 0]['prob_hand_r']

            test = []
            for h in range(0, len(pose_test)):
                pose_test[h] = scale(pose_test[h], 0, 10, True)
                face_test[h] = scale(face_test[h], 0, 10, True)
                hl_test[h] = scale(hl_test[h],  0, 10, True)
                hr_test[h] = scale(hr_test[h], 0, 10, True)

                test_line = np.hstack((pose_test[h], face_test[h],
                                       hl_test[h], hr_test[h]))
                probs_line = np.hstack((np.mean(pose_test_prob[h]), np.mean(face_test_prob[h]),
                                       np.mean(hl_test_prob[h]), np.mean(hr_test_prob[h])))

                test.append(test_line)
                probs_test.append(probs_line)

            test = np.asarray(test)

            Xstd = test.std(axis=0)
            for i in range(Xstd.size):
                if (Xstd[i] == 0):
                    Xstd[i] = 1

            test = (test - test.mean(axis=0)) / Xstd
            inds = [i for i, s in enumerate(all_gloss_gt) if test_gloss in s]
            idx_max = 0.0
            len_kept = 0.0
            for i in inds:
                tt = np.asarray(test)
                gg = all_GT[i]
            #     #
            #     # #norm_gg = (gg[:, 9] - gg[:, 9].mean()) / gg[:, 9].std()
            #     # #norm_tt = (tt[:, 9] - tt[:, 9].mean()) / tt[:, 9].std()
                #c = np.correlate(gg[:, 15], tt[:, 15], "valid")
                c = scipy.signal.fftconvolve(gg[:, np.asarray([9, 15])], tt[:, np.asarray([9, 15])], mode="valid")
                #plt.plot(np.arange(0, len(c)), c)
                #
                #plt.plot(np.arange(0, len(tt[:,15])), tt[:,15])
                #plt.plot(np.arange(0, len(gg[:,15])), gg[:,15])
                #plt.show()
                #plt.close()

                idx_max += c.argmax()
                #print(idx_max)
                len_kept += int(len(gg[:, 15]))

                #test = tt[idx_max:idx_max + len_kept, :]

            #     #
            #     # gg = all_GT[1]
            #     #
            #     # c = np.correlate(gg[:, 9], tt[:, 9], "same")
            #     # plt.plot(np.arange(0, len(c)), c)
            #     # plt.show()
            #     # plt.close()
            #     #
            #     # gg = all_GT[2]
            #     #
            #     # c = np.correlate(gg[:, 9], tt[:, 9], "same")
            #     # plt.plot(np.arange(0, len(c)), c)
            #     # plt.show()
            #     # plt.close()

            idx_max /= len(inds)
            idx_max = 0
            len_kept /= len(inds)
            print(idx_max)
            print(len_kept)
            plt.plot(test[int(idx_max):int(idx_max + len_kept), 15])
            plt.plot(all_GT[inds[0]][:, 15])
            plt.show()
            plt.close()
            all_test.append(np.asarray(test[int(idx_max):int(idx_max + len_kept), :]))
            all_gloss_test.append(test_gloss)
            all_probs_test.append(np.mean(probs_test, axis=0))


    return np.asarray(all_GT), np.asarray(all_test), all_gloss_gt, all_gloss_test, np.asarray(all_probs_test)


def openpose_score(probs, glosses):
    list_set = set(glosses)
    unique_glosses = (list(list_set))
    all_pp = 0.0
    all_pf = 0.0
    all_phl = 0.0
    all_phr = 0.0
    #all_po = 0.0

    for ug in unique_glosses:
        pp = 0.0
        pf = 0.0
        phl = 0.0
        phr = 0.0
        c = 0
        for p, g in zip(probs, glosses):
            if g == ug:
                pp  += p[0]
                pf  += p[1]
                phl += p[2]
                phr += p[3]
                c += 1

        pp /= c
        pf /= c
        phl /= c
        phr /= c
        print(ug + " ----------------")
        print("Confidence Pose: " + str(pp))
        print("Confidence Face: " + str(pf))
        print("Confidence Hand Left: " + str(phl))
        print("Confidence Hand Right: " + str(phr))

        po = (pp + pf + phl + phr) / 4.0

        print("Confidence Overall: " + str(po))
        print("-----------------------")

        all_pp += pp
        all_pf += pf
        all_phl += phl
        all_phr += phr
        #all_po += po

    all_pp = all_pp / len(unique_glosses)
    all_pf = all_pf / len(unique_glosses)
    all_phl = all_phl / len(unique_glosses)
    all_phr = all_phr / len(unique_glosses)
    #all_po = all_po / len(unique_glosses)

    print("--------- FINAL REPORT --------------")
    print("Confidence Pose: " + str(all_pp))
    print("Confidence Face: " + str(all_pf))
    print("Confidence Hand Left: " + str(all_phl))
    print("Confidence Hand Right: " + str(all_phr))

    all_po = (all_pp + all_pf + all_phl + all_phr) / 4.0

    print("Confidence Overall: " + str(all_po))

    return


def run_error_metric(test_dir, gt_dir):
    gloss_list = ['ABEND', 'ABER', 'ERZÄHLEN', 'JAHR', 'VORGESTERN'] #, 'JAHR',  'ABER', 'ERZÄHLEN', 'VORGESTERN']
    GT, test, gloss_gt, gloss_test, test_probs = collect_data(test_dir, gt_dir, gloss_list)

    # hands
    trajectoriesSet = {}
    testSet = {}
    labels = []
    k = GT.shape[0]
    for i in range(k):
        traj = GT[i]
        #trajectoriesSet[(str(i),)] = [traj[:, 168:]]
        trajectoriesSet[(str(i),)] = [traj[:, :]]
        labels.append(gloss_gt[i])

    j = test.shape[0]
    for i in range(k, k + j):
        traj = test[i - k]
        labels.append(gloss_test[i - k])
        testSet[(str(i),)] = [traj[:, :]]

    openpose_score(test_probs, gloss_test)

    cluster_and_plot(trajectoriesSet, testSet, labels, test_probs)



    return

gt_directory = "/home/steph/Documents/Chapter2/Code/smile-tmp/data/iso_path/"
#test_directory = "/home/steph/Documents/Chapter2/Code/smile-tmp/output_openpose/ABER_e10/"
test_directory = "/home/steph/Documents/Chapter2/Code/EVALUATION_ECCV/gloss_to_video/Stoll/puppet/"
run_error_metric(test_directory, gt_directory)