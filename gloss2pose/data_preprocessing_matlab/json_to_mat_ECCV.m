dataset_path = '/home/steph/Documents/Chapter2/Code/smile-tmp/output_openpose';
dataset_path = '/home/steph/Documents/Chapter2/Code/EVALUATION_ECCV/gloss_to_video/Stoll';
save_path = '/home/steph/Documents/Chapter2/Code/smile-tmp/output_openpose';
save_path = '/home/steph/Documents/Chapter2/Code/EVALUATION_ECCV/gloss_to_video/Stoll';

user_folders = dir(dataset_path);
user_folders = user_folders(~ismember({user_folders.name},{'.','..', '.mat'}));

for i = 1:numel(user_folders)
    
    curr_user = [dataset_path '/' user_folders(i).name];
    curr_rec_sess = dir(curr_user);
    curr_rec_sess = curr_rec_sess(~ismember({curr_rec_sess.name},{'.','..', '.mat'}));
    
    for h = 1:numel(curr_rec_sess)
        
        curr_folder = [curr_user, '/', curr_rec_sess(h).name];
        disp(curr_folder)
        curr_save_folder = [save_path, '/', user_folders(i).name, '/', curr_rec_sess(h).name];
        curr_save_file = [curr_save_folder, '/kinect_openpose.mat'];
                
               % if ~exist(curr_save_file, 'file')
                    dummy = zeros(5,5);
                    save(curr_save_file, 'dummy');
                    
                    curr_jsons = dir(curr_folder);
                    curr_jsons =  curr_jsons(~ismember({curr_jsons.name},{'.','..', 'kinect_openpose.mat'}));

                    openpose = [];
                    for ff = 1:numel(curr_jsons)
                        if ~mod(ff, 1000)
                            disp([num2str(ff),'/',num2str(numel(curr_jsons)), ' File: ', curr_folder, '/', curr_jsons(ff).name ])
                        end
                        curr_file = [curr_folder, '/', curr_jsons(ff).name];
                        curr_pose = parse_json(urlread(['file:///', curr_file]));
                        openpose{end+1,1} = curr_pose;
                    end
                    %save(curr_save_file, 'openpose');
                   
                  input = [];
                  pose = [];
                  face = [];
                  hand_l = [];
                  hand_r = [];
                  prob_pose = [];
                  prob_face = [];
                  prob_hand_l = [];
                  prob_hand_r = [];
                  %wrist_r = [];

                  for j = 1 : length(openpose)
                      XYp = openpose{j, 1}{1, 1}.people{1, 1}.pose_keypoints_2d;
                      XYf = openpose{j, 1}{1, 1}.people{1, 1}.face_keypoints_2d;
                      XYl = openpose{j, 1}{1, 1}.people{1, 1}.hand_left_keypoints_2d;
                      XYr = openpose{j, 1}{1, 1}.people{1, 1}.hand_right_keypoints_2d;
                      Pp = XYp(3:3:end);
                      Pp([10, 11, 13, 14]) = [];
                      Pf = XYf(3:3:end);
                      Pl = XYl(3:3:end);
                      Pr = XYr(3:3:end);
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
                      prob_pose = [prob_pose; cell2mat(Pp)];
                      prob_face = [prob_face; cell2mat(Pf)];
                      prob_hand_l = [prob_hand_l; cell2mat(Pl)];
                      prob_hand_r = [prob_hand_r; cell2mat(Pr)];
                      %wrist_r =[wrist_r;XYwr];


                  end


                   input(1).pose = pose;
                   input(1).face = face;
                   input(1).hand_l = hand_l;
                   input(1).hand_r = hand_r;
                   input(1).prob_pose = prob_pose;
                   input(1).prob_face = prob_face;
                   input(1).prob_hand_l = prob_hand_l;
                   input(1).prob_hand_r = prob_hand_r;
                   
                   %input(count).wrist_r = wrist_r;
                   %input(count).path = strcat(signer_info(L).folder, '/', signer_info(L).name, '/color.mp4');
                    %end
                   save(strcat(curr_save_folder,'SMILE_keypoints.mat'), 'input');
    end
end