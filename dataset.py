import os
import cv2

class Dataset(object):
    
    def __init__(self):
        self.dataset_name = 'tiny_pose_dataset'
        self.num_kps = 14
        self.kps_names = ['head_top',   'head_bottom', 
                          'l_shoulder', 'r_shoulder',
                          'l_elbow',    'r_elbow', 
                          'l_wrist',    'r_wrist',
                          'l_hip',      'r_hip', 
                          'l_knee',     'r_knee', 
                          'l_ankle',    'r_ankle']

        self.kps_symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
        self.kps_lines    = [(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), 
                             (5, 7), (7, 9), (11, 13), (5, 6), (11, 12)]

        self.dataset_path = 'tiny_pose_dataset'
        self.image_path   = os.path.join(self.dataset_path, 'Images')
        self.label_path   = os.path.join(self.dataset_path, 'Annotations')
    

    def load_frame_data(self, task = 'train'):
        if task in ['train', 'val']:
            image_path = os.path.join(self.image_path, task)
            label_path = os.path.join(self.label_path, task)
        else:
            assert(task in ['train', 'val']), "please type \'train\' or \'val\'"
            
        frame_data = []
        folder_lst = sorted([x for x in os.listdir(image_path) if 
                             os.path.isdir(os.path.join(image_path, x))])
        
        for folder_name in folder_lst:
            # gather image_path and joints from start to end of video
            label_name = folder_name.split('_')[1]
            image_path_lst = []
            joints_lst     = []
            
            image_folderPath = os.path.join(image_path, folder_name)
            label_folderPath = os.path.join(label_path, folder_name)

            image_lst = sorted([y for y in os.listdir(image_folderPath) if 
                                os.path.splitext(y)[-1] in ['.jpg', '.png']])
            
            for image_name in image_lst:
                label_name = image_name.split('.')[0] + '.txt'
                imagefilePath = os.path.join(image_folderPath, image_name)
                labelfilePath = os.path.join(label_folderPath, label_name)
              
                with open(labelfilePath, 'rt') as f:
                    for line in f:
                        label_read_info = [int(s) for s in line.rsplit()[:42]]
                        
                data = dict(image_path = imagefilePath,
                            joints     = label_read_info)
                frame_data.append(data)
        
        return frame_data
        
