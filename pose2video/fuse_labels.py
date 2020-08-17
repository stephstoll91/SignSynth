import os
from PIL import Image
import numpy as np

hand_labels_path = "/media/Depo/steph/BSL_Translations/Dynavis/label_hands/"
face_labels_path = "/media/Depo/steph/BSL_Translations/Dynavis/label_face/"
pose_labels_path = "/media/Depo/steph/BSL_Translations/Dynavis/label/"

fused_labels_path = "/media/Depo/steph/BSL_Translations/Dynavis/label_all/"
fused_test_path = "/media/Depo/steph/BSL_Translations/Dynavis/dataset/test_label/"
fused_train_path = "/media/Depo/steph/BSL_Translations/Dynavis/dataset/train_label/"
for (root, dirs, files) in os.walk(pose_labels_path):
    for f in files:
        pose_file = os.path.join(pose_labels_path, f)
        hand_file = os.path.join(hand_labels_path, f)
        face_file = os.path.join(face_labels_path, f)

        pp = Image.open(pose_file)
        arrp = np.array(pp)

        hh = Image.open(hand_file)
        arrh = np.array(hh)

        ff = Image.open(face_file)
        arrf = np.array(ff)

        arr1 = np.zeros((720, 1280), dtype=np.uint8)
        for x in range(1280):
            for y in range(720):
                pose_p = arrp[y][x]
                face_p = arrf[y][x]
                hand_p = arrh[y][x]
                ml = [pose_p, face_p, hand_p]
                m = max(ml)
                arr1[y][x] = m
        #arr1 = arrp + arrh
        #arr = arr1 + arrf

        img = Image.fromarray(arr1)

        if "1_Overview" in f:
            img.save(os.path.join(fused_test_path, f))
        else:
            img.save(os.path.join(fused_train_path, f))



