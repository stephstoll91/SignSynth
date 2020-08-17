import os

path = "/media/Depo/steph/BSL_Translations/Dynavis/videos/"
img_path = "/media/Depo/steph/BSL_Translations/Dynavis/img/"
png_path = "/media/Depo/steph/BSL_Translations/Dynavis/label/"

if not os.path.exists(img_path):
    os.makedirs(img_path)

videos = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


for v in videos:
    iname = img_path + os.path.splitext(v)[0] + "_%12d.png"
    vname = path + v
    os.system("ffmpeg -i " + vname + " -r 25 -f image2 " + iname)

frames = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

for f in frames:
    fname = img_path + f
    lname = png_path + os.path.splitext(f)[0] + "_keypoints.png"
    if not os.path.isfile(lname):
        os.remove(fname)