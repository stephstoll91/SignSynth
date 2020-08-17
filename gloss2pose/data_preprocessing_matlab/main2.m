dataset_path = '/home/steph/Documents/Chapter2/Code/smile-tmp/output_openpose';
save_path = '/home/steph/Documents/Chapter2/Code/smile-tmp/output_openpose';

user_folders = dir(dataset_path);
user_folders = user_folders(~ismember({user_folders.name},{'.','..'}));

for i = 1:numel(user_folders)
    
    curr_user = [dataset_path '/' user_folders(i).name];
    curr_rec_sess = dir(curr_user);
    curr_rec_sess = curr_rec_sess(~ismember({curr_rec_sess.name},{'.','..'}));
    
    for j = 1:numel(curr_rec_sess)
        
        curr_folder = [curr_user, '/', curr_rec_sess(j).name];
        disp(curr_folder)
        curr_save_folder = [save_path, '/', user_folders(i).name, '/', curr_rec_sess(j).name];
        curr_save_file = [curr_save_folder, '/kinect_openpose.mat'];
                
               % if ~exist(curr_save_file, 'file')
                    dummy = zeros(5,5);
                    save(curr_save_file, 'dummy');
                    
                    curr_jsons = dir(curr_folder);
                    curr_jsons =  curr_jsons(~ismember({curr_jsons.name},{'.','..', 'kinect_openpose.mat'}));

                    people = [];
                    for ff = 1:numel(curr_jsons)
                        if ~mod(ff, 1000)
                            disp([num2str(ff),'/',num2str(numel(curr_jsons)), ' File: ', curr_folder, '/', curr_jsons(ff).name ])
                        end
                        curr_file = [curr_folder, '/', curr_jsons(ff).name];
                        curr_pose = parse_json(urlread(['file:///', curr_file]));
                        people{end+1,1} = curr_pose;
                    end
                    save(curr_save_file, 'people');
                %end
    end
end