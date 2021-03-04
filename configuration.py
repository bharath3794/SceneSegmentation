import glob
from collections import namedtuple


class Config:
    def __init__(self, n_shots=4, n_seq=10, batch_size=8, learning_rate=1e-2, num_epochs=2, cp_dir='', save_dir='', plots_dir=''):
        self.n_shots = n_shots
        self.n_seq = n_seq
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.cp_dir = cp_dir
        self.save_dir = save_dir
        self.plots_dir = plots_dir
        self.learning_rate = learning_rate
        self.data_folder = "./data/" # Change to your data directory path
        self.movies_list = sorted(glob.glob(self.data_folder+"*.pkl"))
        self.feature_names = {'place', 'cast', 'action', 'audio', 
                              'scene_transition_boundary_ground_truth'}
        self.modelConfig = namedtuple('modelConfig', 
                                       ['name', 'sim_channel', 'place_feat_dim', 'cast_feat_dim',
                                        'act_feat_dim', 'aud_feat_dim', 'aud', 'bidirectional',
                                        'lstm_hidden_size', 'ratio'])
        self.movies_ref_dict = {i: self.movies_list[i][-13:-4] for i in range(len(self.movies_list))}
        self.train_set_idx = int(len(self.movies_list)*0.9)


        # Parameters required to use AnyiRao's Model
            # github (https://github.com/AnyiRao/SceneSeg) 
        self.shot_num = n_shots
        self.seq_len = n_seq
        self.model = self.modelConfig(
                        name='LGSS',
                        sim_channel=512,  # dim of similarity vector
                        place_feat_dim=2048,
                        cast_feat_dim=512,
                        act_feat_dim=512,
                        aud_feat_dim=512,
                        aud=dict(cos_channel=512),
                        bidirectional=True,
                        lstm_hidden_size=512,
                        ratio=[0.5, 0.2, 0.2, 0.1]
                        )
