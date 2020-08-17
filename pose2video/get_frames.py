import os
import shutil

source_dir = "/home/steph/Documents/Chapter2/Code/p2pTEST/remote/pix2pixHD/results/ECCV/more_singles_seqs_for_video/"
target_dir = "/home/steph/Documents/Chapter2/Code/p2pTEST/remote/pix2pixHD/results/ECCV/more_singles_seqs_frame_for_video/"

for root, dirs, files in os.walk(source_dir):
    for dir in dirs:
        if os.path.isdir(source_dir + dir):
            seqs = [d for d in os.listdir(source_dir + dir) if os.path.isdir(os.path.join(source_dir + dir, d))]
            for s in seqs:
                tar = os.path.join(target_dir + dir, s)
                os.system('mkdir -p ' + tar)
                ss = os.path.join(source_dir + dir, s, 'train_mollica_small', 'test_latest', 'images')
                pics = [p for p in os.listdir(ss) if (os.path.isfile(os.path.join(ss, p)) and 'synthesized' in p)]
                for p in pics:
                    p_source = os.path.join(ss, p)
                    os.system('cp ' + p_source + ' ' + os.path.join(tar, p))

for root, dirs, files in os.walk(target_dir):
    for dir in dirs:
        if os.path.isdir(target_dir + dir):
            tt = target_dir + dir + '/'
            seqs = [d for d in os.listdir(tt) if os.path.isdir(os.path.join(tt, d))]
            for s in seqs:
                #os.system('cd ' + tt + s)
                os.system('ffmpeg -f image2 -pattern_type glob -i \'' + tt + s + '/*.jpg\' ' + dir + s + '.mp4')
                os.system('cp -v ' + dir + s + '.mp4 ' + '/home/steph/Documents/Chapter2/Code/EVALUATION_ECCV/VIDEO/')
                #os.system('cd ..')
