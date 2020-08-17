# coding=utf-8
import numpy as np
import math
import scipy.io as sio
#from scipy.spatial import procrustes
from os import listdir
from os.path import isfile, join
#import matplotlib.pyplot as plt

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
    ref_pose = np.load('/vol/vssp/cvpnobackup/Steph_tfer/smile-tmp/Mollica-ref-pose.npy')[0]
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

base_dir = "/vol/vssp/cvpnobackup/Steph_tfer/pycharm_projects/pose_regressor/data/iso_path/"

data_mats = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
class_file = open('/vol/vssp/cvpnobackup/Steph_tfer/smile-tmp/vocab_classes.txt', "w")

vc = 0
for d in data_mats:
    voc = d.split("_")[-2]
    class_file.write(str(vc) + " " + voc + "\n")
    vc += 1
class_file.close()

inputs = []
outputs = []
labels_in = []
labels_out = []
path_in = []
path_out = []
count = 0
for f in data_mats:
    data = sio.loadmat(base_dir + f)
    input = data['input']
    one_hot = np.zeros(len(data_mats))
    one_hot[count] = 1.0

    _,k = input.shape

    for l in range(3, k):

        normed_input = normalise(input, l)
        pose = normed_input[0, l]['pose']
        face = normed_input[0, l]['face']
        hl = normed_input[0, l]['hand_l']
        hr = normed_input[0, l]['hand_r']

        for h in range(len(pose)-2):
            framep = np.reshape(pose[h], (-1, 2))
            facep = np.reshape(face[h], (-1, 2))
            hlp = np.reshape(hl[h], (-1, 2))
            hrp = np.reshape(hr[h], (-1, 2))
            # plt.figure()
            # plt.scatter(framep[:, 0], framep[:, 1])
            # plt.scatter(facep[:, 0], facep[:, 1])
            # plt.scatter(hlp[:, 0], hlp[:, 1])
            # plt.scatter(hrp[:, 0], hrp[:, 1])
            # plt.gca().invert_yaxis()
            # plt.savefig('./norms/' + str(h) + '.png')
            # plt.close()

            if h == 0:
                input_line = np.hstack((pose[h], face[h], hl[h], hr[h], pose[h] - pose[h], face[h] - face[h], hl[h] - hl[h], hr[h] - hr[h]))

            else:
                input_line = np.hstack((pose[h], face[h], hl[h], hr[h], pose[h] - pose[h-1], face[h] - face[h-1], hl[h] - hl[h-1], hr[h] - hr[h-1]))

            output_line = np.hstack((pose[h + 1], face[h + 1], hl[h + 1], hr[h + 1], pose[h + 1] - pose[h], face[h + 1] - face[h], hl[h + 1] - hl[h], hr[h + 1] - hr[h]))

            label_line_in = np.hstack((one_hot, input_line))
            label_line_out = np.hstack((one_hot, output_line))

            inputs.append(input_line)
            outputs.append(output_line)
            labels_in.append(label_line_in)
            labels_out.append(label_line_out)
            #path_in.append(path[0] + "/" + str(h))
            #path_out.append(path[0] + "/" + str(h+1))

    count += 1

# np.save("/home/steph/Documents/Chapter2/Code/smile-tmp/data/normed_inputs.npy", inputs)
# np.save("/home/steph/Documents/Chapter2/Code/smile-tmp/data/normed_outputs.npy", outputs)
# np.save("/home/steph/Documents/Chapter2/Code/smile-tmp/data/normed_labels_in.npy", labels_in)
# np.save("/home/steph/Documents/Chapter2/Code/smile-tmp/data/normed_labels_out.npy", labels_out)

np.savez("/vol/vssp/cvpnobackup/Steph_tfer/smile-tmp/data/new_normed_smile_data_input_train_eval_test.npz",
         input=np.asarray(inputs), output=np.asarray(outputs), label_in=np.asarray(labels_in),
         label_out=np.asarray(labels_out), path_in=np.asarray(path_in), path_out=np.asarray(path_out))
