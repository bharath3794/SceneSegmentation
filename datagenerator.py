import torch
import pickle
import numpy as np


class CustomDataGenerator(torch.utils.data.Dataset):
    def __init__(self, config, features_idxs):
        self.movies_dir = config.data_folder
        self.movies_lst = config.movies_list
        self.n_shots = config.n_shots
        self.feature_names = config.feature_names
        self.features_idxs = features_idxs
    def __len__(self):
        return len(self.features_idxs)
    def __getitem__(self, i):
        for movie_idx, feature_idx in self.features_idxs[i].items():
            with open(self.movies_lst[int(movie_idx.split('.')[0])], 'rb') as f:
                cur_movie = pickle.load(f)
            if 'place' in self.feature_names:
                place_features = self.get_features('place', cur_movie, feature_idx)
            else:
                place_features = [[] for i in range(len(feature_idx))]
            if 'cast' in self.feature_names:
                cast_features = self.get_features('cast', cur_movie, feature_idx)
            else:
                cast_features = [[] for i in range(len(feature_idx))]
            if 'action' in self.feature_names:
                action_features = self.get_features('action', cur_movie, feature_idx)
            else:
                action_features = [[] for i in range(len(feature_idx))]
            if 'audio' in self.feature_names:
                audio_features = self.get_features('audio', cur_movie, feature_idx)
            else:
                audio_features = [[] for i in range(len(feature_idx))]
            if 'scene_transition_boundary_ground_truth' in self.feature_names:
                labels, reference_idxs =  self.get_features('scene_transition_boundary_ground_truth', cur_movie, feature_idx, movieIdx=movie_idx, is_label=True)            
        return place_features, cast_features, action_features, audio_features, labels, reference_idxs
            
    def get_features(self, key, cur_movie, feature_idx, movieIdx='', is_label=False):
        '''Dataset Preparation: Create a feature for the given sample which consists of indexes
        Args:
        	key: Select the feature you are looking for: Place, Cast, Action, Audio features
        	feature_idx: sample feature consisting of indexes to consider from the pickle file
        				 of the movie in data directory
        	is_label: if True, we need to return ground truths to use them as target while training
        Returns:
        	Feature of the given key and the current movie			
        '''
        feature = []
        for i in range(len(feature_idx)):
            if is_label:
                is_shot_boundary = 1 if cur_movie[key][feature_idx[i][(self.n_shots//2)-1]] else 0
                feature.append(is_shot_boundary)
                if i==0:
                    reference_idx = feature_idx[i][(self.n_shots//2)-1]
            else:
                feature.append([])
                for j in range(len(feature_idx[i])):
                    feature[-1].append(cur_movie[key][feature_idx[i][j]])
        if is_label:
            return torch.from_numpy(np.array(feature)).type(torch.LongTensor), (movieIdx, reference_idx)
        feature = [torch.stack(feature[i]) for i in range(len(feature))]
        return torch.stack(feature)