root = "/home/steph/Documents/Chapter2/Code/smile-tmp/output_openpose/";

vocab_dir = "/vol/vssp/smile/SMILE_ISO/I50_Juon-Andreas_L1/";
vocab_list = [];
vocab_info = dir(vocab_dir);
for v = 3 : length(vocab_info)
    dir_name = vocab_info(v).name;
    ss = strsplit(dir_name,"_");
    vocab_list = [vocab_list; strcat(string(ss{1}), "_")];
end
vl = unique(vocab_list);

root_info = dir(root);

for vc = 1 : length(vl)
    input = [];
    out_gt = [];

    count = 1;
    for K = 3 : length(root_info)
      thisdir = root_info(K).name;
      if strfind(root_info(K).name, 'I20_Correia-Jessica_L1')

      else
          signer_info = dir(strcat(root_info(K).folder, "/", thisdir));
      for L = 3 : length(signer_info)
          if strfind(signer_info(L).name, vl{vc}) == 1
              f = strcat(signer_info(L).folder, "/", signer_info(L).name, "/kinect_openpose.mat")
              load(f);
              pose = [];
              face = [];
              hand_l = [];
              hand_r = [];
              %wrist_r = [];

              for j = 1 : length(openpose)
                  XYp = openpose{j, 1}.pose_keypoints;
                  XYf = openpose{j, 1}.face_keypoints;
                  XYl = openpose{j, 1}.hand_left_keypoints;
                  XYr = openpose{j, 1}.hand_right_keypoints;

                  %pose
                  XYp(3:3:end) = [];
                  B =reshape(XYp,2,[]);
                  XYwr = reshape(B(:,5), 1,[]);
                  B(:,[10, 11, 13, 14]) = [];
                  XYp =reshape(B,1,[]);

                  %face
                  XYf(3:3:end) = [];

                  %left hand
                  XYl(3:3:end) = [];

                  %right hand
                  XYr(3:3:end) = [];

                  %for COCO pose, ignoring legs:
                  XYp = cell2mat(XYp);

                  XYf = cell2mat(XYf);

                  XYl = cell2mat(XYl);

                  XYr = cell2mat(XYr);

                  XYwr = cell2mat(XYwr);

                  pose = [pose;XYp];
                  face = [face;XYf];
                  hand_l = [hand_l;XYl];
                  hand_r = [hand_r;XYr];
                  %wrist_r =[wrist_r;XYwr];


              end


               input(count).pose = pose;
               input(count).face = face;
               input(count).hand_l = hand_l;
               input(count).hand_r = hand_r;
               %input(count).wrist_r = wrist_r;
               input(count).path = strcat(signer_info(L).folder, '/', signer_info(L).name, '/color.mp4');
               count = count + 1;
         end
      end
      end

    end
    %save('/vol/vssp/smile/Steph/pycharm_projects/pose_regressor/smile_data.mat', 'input', 'out_gt');
    save(strcat('/vol/vssp/smile/Steph/pycharm_projects/pose_regressor/data/iso_path/smile_data_input_', vl{vc},'.mat'), 'input');
end
