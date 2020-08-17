import os

path = "/media/Depo/steph/BSL_Translations/Dynavis/"

onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print onlyfiles
for f in onlyfiles:
	if f.endswith(".mp4"):
		os.system("./build/examples/openpose/openpose.bin --video " + path + f + " --write_json /media/Depo/steph/BSL_Translations/Dynavis/json/ --output_resolution -1x-1 --hand --face")

		
