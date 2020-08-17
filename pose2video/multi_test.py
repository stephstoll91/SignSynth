### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from distutils.dir_util import copy_tree

_data_path = '/vol/vssp/cvpnobackup/Steph_tfer/smile-tmp/test/new_normed_BC30_h64_no_eval/e5maps/'
data_out_path = "/vol/vssp/cvpnobackup/Steph_tfer/pycharm_projects/pix2pixHD/results/ECCV/results/"
tmp_dir = "/vol/vssp/cvpnobackup/Steph_tfer/pycharm_projects/pix2pixHD/results/ECCV/tmp/test_label/"
dataset_root = "/vol/vssp/cvpnobackup/Steph_tfer/pycharm_projects/pix2pixHD/results/ECCV/tmp/"

if not os.path.exists(data_out_path):
    os.makedirs(data_out_path)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if os.path.isdir('/vol/vssp/cvpnobackup/Steph_tfer/pycharm_projects/pix2pixHD/results/ECCV/tmp/test_label/*'):
    os.system("rm -r /vol/vssp/cvpnobackup/Steph_tfer/pycharm_projects/pix2pixHD/results/ECCV/tmp/test_label/*")
for root, dirs, files in os.walk(_data_path):
    for d in dirs:
        if not d.startswith('.'):
            if os.path.isdir(os.path.join(_data_path, d)):
                gloss_path = _data_path + d + '/'
                print(gloss_path)
                seqs = [d for d in os.listdir(gloss_path) if os.path.isdir(os.path.join(gloss_path, d))]
                for seq in seqs:
                    print(seq)
                    seq_path = gloss_path + seq + '/'
                    print(seq_path)
                    copy_tree(seq_path, tmp_dir)
                    seq_out_dir = data_out_path + d + '/' + seq + '/'
                    print(seq_out_dir)
                    opt = TestOptions().parse(save=False)
                    opt.nThreads = 1  # test code only supports nThreads = 1
                    opt.batchSize = 1  # test code only supports batchSize = 1
                    opt.serial_batches = True  # no shuffle
                    opt.no_flip = True  # no flip
                    opt.dataroot = dataset_root
                    opt.results_dir = seq_out_dir

                    data_loader = CreateDataLoader(opt)
                    dataset = data_loader.load_data()
                    visualizer = Visualizer(opt)
                    # create website
                    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
                    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
                        opt.name, opt.phase, opt.which_epoch))

                    # test
                    if not opt.engine and not opt.onnx:
                        model = create_model(opt)
                        if opt.data_type == 16:
                            model.half()
                        elif opt.data_type == 8:
                            model.type(torch.uint8)

                        if opt.verbose:
                            print(model)
                    else:
                        from run_engine import run_trt_engine, run_onnx

                    for i, data in enumerate(dataset):
                        if i >= opt.how_many:
                            break
                        if opt.data_type == 16:
                            data['label'] = data['label'].half()
                            data['inst'] = data['inst'].half()
                        elif opt.data_type == 8:
                            data['label'] = data['label'].uint8()
                            data['inst'] = data['inst'].uint8()
                        if opt.export_onnx:
                            print ("Exporting to ONNX: ", opt.export_onnx)
                            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
                            torch.onnx.export(model, [data['label'], data['inst']],
                                              opt.export_onnx, verbose=True)
                            exit(0)
                        minibatch = 1
                        if opt.engine:
                            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
                        elif opt.onnx:
                            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
                        else:
                            generated = model.inference(data['label'], data['inst'])

                        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                               ('synthesized_image', util.tensor2im(generated.data[0]))])
                        img_path = data['path']
                        print('process image... %s' % img_path)
                        visualizer.save_images(webpage, visuals, img_path)

                        webpage.save()

                        # delete files in tmp
                        os.system(
                            "rm -r /vol/vssp/cvpnobackup/Steph_tfer/pycharm_projects/pix2pixHD/results/ECCV/tmp/test_label/*")

