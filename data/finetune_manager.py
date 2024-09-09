from .pikey_manager import BaseLoader
import torch
import numpy as np
import os

class FinetuneLoader(BaseLoader):
    def __init__(self, params):
        super(FinetuneLoader, self).__init__(params)
        # pin_memory = 'tiered' not in params.dataset
        # 'training': True,
        # 'fewshot': False,
        # 'image_size': 84,
        # 'fewshot_params': None,
        # 'fix': False,
        # 'pin_memory': False,
        # 'data_file': '',
        # 'batch_size': 128,
        # 'shuffle': True,
        # 'aug': True

        self.configures = self.create_config([
            # config['contrast'], config['nview']
            {'name': 'S_Base_FS', 'training': True, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'base.json')},
            {'name': 'S_Base_FS_Fix', 'training': True, 'fewshot': True,'fix':True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'base.json')},
            {'name': 'S_Base', 'training': True, 'fewshot': False,'batch_size':16,
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'base.json')},
            {'name': 'S_Val', 'training': False, 'fewshot': False,
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'val.json')},
            # {'name': 'Weighted_S_Base_FS', 'training': True, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.train_episode],
            #  'pin_memory': False, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'base.json'),'weighted_samper':True},
            # {'name': 'S_Val_FS', 'training': False, 'aug': False, 'fix': True, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.test_episode],
            #  'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'val.json')},
            # {'name': 'S_Novel_FS', 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.test_episode],
            #  'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'novel.json')},
            {'name': 'A_Base_FS', 'training': True, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True, 'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_{self.params.target_num_label}.json'},
            {'name': 'A_Base_FS_Fix', 'training': True, 'fewshot': True,'fix':True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True,
             'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_{self.params.target_num_label}.json'},
            {'name': 'A_Base_FS_Strong', 'strong_aug':True, 'training': True, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True,
             'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json'},
            # {'name': 'A_Base_FS2', 'training': True, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.train_episode],
            #  'pin_memory': True,
            #  'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json'},
            {'name': 'A_Base', 'training': False, 'fewshot': False,
             'pin_memory': True,
             'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json'},
            {'name': 'A_Base_Full_FS', 'training': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True,
             'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'base.json')},
            # {'name': 'A_Novel_Full_FS', 'training': False, 'aug': False, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.train_episode],
            #  'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'novel.json')},
            {'name': 'A_Val_Full_FS', 'training': False, 'aug': False, 'fix': True, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'val.json')},
            # {'name': 'A_Visualize', 'training': False, 'aug': False, 'fix': True, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.train_episode],
            #  'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'val.json'),'return_gt':True},
            # {'name': 'S_Visualize', 'training': False, 'aug': False, 'fix': True, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.test_episode],
            #  'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'novel.json'),'return_gt':True},
            # {'name': 'A_Meta', 'training': True, 'fewshot': True,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.train_episode],
            #  'pin_memory': True, 'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json'},
        ])
        self.setup()

class VisualizeLoader(BaseLoader):
    def __init__(self, params):
        super(VisualizeLoader, self).__init__(params)
        self.configures = self.create_config([
            {'name': 'A_Visualize', 'return_gt':True, 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True, 'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json','return_gt':True},
            {'name': 'A_Visualize_Novel', 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True,
              'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'novel.json'),
             'return_gt': True},
            {'name': 'S_Visualize', 'return_gt':True,'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.test_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'base.json'),'return_gt':True},

            {'name': 'S_Visualize_Novel', 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.test_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'novel.json'),
             'return_gt': True},
        ])
        self.setup()

class TestLoader(BaseLoader):
    def __init__(self, params):
        super(TestLoader, self).__init__(params)
        self.configures = self.create_config([
            {'name': 'T_Novel_FS', 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.test_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'novel.json')},
            {'name': 'T_Base_FS', 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.test_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'base.json')},
            {'name': 'T_Aux_FS', 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.test_episode],
             'pin_memory': True,  'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json'},
            {'name': 'S_Novel_FS', 'training': False, 'aug': False, 'fix': False, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.test_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'novel.json')},
        ])
        self.setup()

class Analysis(BaseLoader):
    def __init__(self, params):
        super(Analysis, self).__init__(params)
        self.configures = self.create_config([
            {'name': 'S_Base', 'training': False, 'fewshot': False, 'batch_size': 64,
             'pin_memory': False, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'base.json')},
            {'name': 'S_Base_FS', 'training': False, 'fewshot': True, 'fix':False,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'base.json')},
            {'name': 'A_Base_Full', 'training': False, 'fewshot': False, 'batch_size': 64, 'aug': False,
             'pin_memory': False,
             'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'base.json')},
            {'name': 'S_Novel', 'training': False, 'aug': False, 'fix': False, 'fewshot': False,'batch_size': 64
                ,'pin_memory': False, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'novel.json')},
            {'name': 'A_Novel', 'training': False, 'fewshot': False, 'batch_size': 64, 'aug': False,
             'pin_memory': False,'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'novel.json')},
            {'name': 'A_Base', 'training': False, 'fewshot': False, 'batch_size': 64,
             'pin_memory': False,
             'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json'},
            # {'name': 'A_Base_Full_FS', 'training': False, 'fewshot': True,'fix':False,
            #  'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
            #                     self.params.train_episode],
            #  'pin_memory': True,
            #  'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'base.json')},
            {'name': 'A_Base_FS', 'training': False, 'fewshot': True,'fix':False, 'aug': False,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True,
             'data_file': f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_{self.params.target_set}_5.json'},

             # 'data_file': os.path.join(self.params.data_dir,self.params.target_set,'base.json')},
            {'name': 'S_Novel_FS', 'training': False, 'aug': False, 'fix': True, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.test_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, 'miniImagenet', 'novel.json')},
            {'name': 'A_Novel_FS', 'training': False, 'aug': False,'fix':True, 'fewshot': True,
             'fewshot_params': [self.params.train_n_way, self.params.n_shot, 16,
                                self.params.train_episode],
             'pin_memory': True, 'data_file': os.path.join(self.params.data_dir, self.params.target_set, 'novel.json')},
        ])
        self.setup()