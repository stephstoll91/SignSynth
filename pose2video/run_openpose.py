dir_of_frames = "/home/steph/Documents/Chapter2/Code/p2pTEST/remote/pix2pixHD/results/ECCV/result_frames/"
dir_of_openpose = "/home/steph/Documents/Chapter2/Code/p2pTEST/remote/pix2pixHD/results/ECCV/result_openpose/"

for root, dirs, files in os.walk(dir_of_frames):
    for dir in dirs:
        if os.path.isdir(dir_of_frames + dir):
            seqs = [d for d in os.listdir(source_dir + dir) if os.path.isdir(os.path.join(source_dir + dir, d))]
            for s in seqs:
                tar = os.path.join(dir_of_openpose + dir, s)
                os.system("")