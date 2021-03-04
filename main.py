import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score


# Importing my files
from configuration import Config
from datagenerator import CustomDataGenerator
from rao_model.model import *
from elv_ml_challenge.evaluate_sceneseg import *

# * This global variable is used for reconstructing the predicted output
#	of the test set internally from the test function
# Dict Keys: imdbID of movies from Test set
# Dict Values: A tensor of shape 'scene_transition_boundary_prediction'
test_boundary_prediction = {}

def get_data_indexes(data, config, movie_id):
	'''Dataset Preparation: Form indexes of each movie accordingly
	Args:
		data: List of indexes with total no. of shots in that movie
		config: Configuration Paramters
		movie_id: Movie ID is the index of that movie from list of files 
				  in the data directory
	Returns:
		   rslt: A list of dictionaries with sample number as key (depends on shot length and
		   sequence length to consider) and sample indexes as its value for the given
		   Movie ID
	'''
	if config.n_shots > len(data):
		raise Exception("n_shots > len(features), Please decrease n_shots.")
	rslt = []
	lst = []
	for k in range(len(data)//config.n_seq):
		checkSeqLen = 0
		temp = []
		for i in range(k*config.n_seq, len(data)-config.n_shots+1):
			if checkSeqLen == config.n_seq:
				break
			temp.append([])
			for j in range(i, i+config.n_shots):
				temp[-1].append(data[j])
			checkSeqLen += 1
		if len(temp)<config.n_seq:
			break
		lst.append(temp)
	for p in range(len(lst)):
		rslt.append({"{}.{}".format(movie_id, p):lst[p]})
	return rslt


def get_features_idxs(config):
	'''Dataset Preparation: Prepare train (later split to train and val) 
	   and test set indices for all the movies in data directory
	 Args:
	 	 config: Configuration parameters
	 Returns:
	 	train_features_idxs, test_features_idxs:
	 	Train and Test indexes for all the movies in data directory
	'''
	global test_boundary_prediction
	files_list = config.movies_list
	train_features_idxs = []
	test_features_idxs = []
	for i in range(len(files_list)):
		with open(files_list[i], 'rb') as f:
			cur_movie = pickle.load(f)
		# Initial 90% of movies from the data directory are considered for train set
		# Last 10% of movies are considered for test set
		if i < config.train_set_idx:
			train_features_idxs.extend(get_data_indexes(list(range(len(cur_movie['place']))), config, i))
		else:
			# * Fill test_boundary_prediction dictionary with key as imdbID of current movie 
			# from Test set and value as tensor of shape 'scene_transition_boundary_prediction'
			# filled with values '0.'
			test_boundary_prediction[config.movies_ref_dict[i]] = torch.empty(cur_movie['scene_transition_boundary_prediction'].shape[0]).fill_(0.)
			test_features_idxs.extend(get_data_indexes(list(range(len(cur_movie['place']))), config, i))
	return train_features_idxs, test_features_idxs

# This global variable is used for noting the current training iteration number 
# of all the epochs combined
train_iter = 0
def train(config, model, device, train_loader, optimizer, scheduler, criterion, writer, epoch):
	'''Trains the model based on the given train data loader
	Prints:
		Train Loss for every 512 samples from the created train dataset
	Updates:
		Train Loss for every 512 samples is written to the SummaryWriter object 
		for visualization purposes
	Returns:
		Average Train Loss for current epoch
	'''
	global train_iter
	model.train()
	losses = []
	for batch_idx, (data_place, data_cast, data_act, data_aud, target, ref_idxs) in enumerate(train_loader):
		data_place = data_place.to(device) if 'place' in config.feature_names else []
		data_cast  = data_cast.to(device) if 'cast' in config.feature_names else []
		data_act   = data_act.to(device) if 'act' in config.feature_names else []
		data_aud   = data_aud.to(device) if 'aud' in config.feature_names else []
		target = target.view(-1).to(device)

		optimizer.zero_grad()

		output = model(data_place, data_cast, data_act, data_aud)

		output = output.view(-1, 2)

		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		train_iter += 1
		# For every 512 samples print and update instead of doing it for every batch
		if batch_idx % (512//config.batch_size) == 0:
			# Consider average train loss of last 512 samples only after appending loss in each batch
			writer.add_scalar('train/loss', round(np.mean(losses[-(512//config.batch_size):]), 4), train_iter)
			# Print only current batch loss (For ex: 0th, 512th, 1024th samples batches are printed)
			print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			    epoch, int(batch_idx * len(data_place)), len(train_loader.dataset),
			    100. * batch_idx / len(train_loader), loss.item()))
	# Average Train Loss of current epoch
	train_loss = round(np.mean(losses), 4)
	# Make Learning rate scheduler to take a step
	scheduler.step()
	return train_loss


# This global variable is used for noting the current val iteration number 
# of all the epochs combined
val_iter = 0
def test(config, model, device, test_loader, criterion, writer, epoch, is_test=True):
	'''Val/Tests the model based on the given val and test data loaders
	Args:
		is_test: if True, perform testing else, perform validating
	Prints:
		Val Loss for every 512 samples from the created val dataset
	Updates: (Only for Validation but not for testing)
		Val Loss for every 512 samples is written to the SummaryWriter object 
		for visualization purposes
	Returns:
		Average Val/Test Loss for current epoch
	'''
	global val_iter, test_boundary_prediction
	model.eval()
	losses = []
	with torch.no_grad():
		for batch_idx, (data_place, data_cast, data_act, data_aud, target, ref_idxs) in enumerate(test_loader):
			data_place = data_place.to(device) if 'place' in config.feature_names else []
			data_cast  = data_cast.to(device) if 'cast' in config.feature_names else []
			data_act   = data_act.to(device) if 'act' in config.feature_names else []
			data_aud   = data_aud.to(device) if 'aud' in config.feature_names else []
			target = target.view(-1).to(device)

			output = model(data_place, data_cast, data_act, data_aud)
			output = output.view(-1, 2)
			
			loss = criterion(output, target)
			losses.append(loss.item())
			# If testing, we need to reconstruct our predictions to evaluate mAP, mIoU
			if is_test:
				output = F.softmax(output, dim=1)
				pred = output[:, 1]
				k = 0
				for i in range(len(ref_idxs[0])):
					key = config.movies_ref_dict[int(ref_idxs[0][i].split('.')[0])]
					for j in range(ref_idxs[1][i], ref_idxs[1][i]+config.n_seq):
						test_boundary_prediction[key][j] = pred[k]
						k += 1
				assert k == len(pred)
			else:
				val_iter += 1
			mode = '\tTest' if is_test else '\tVal'
			if batch_idx % (512//config.batch_size)==0:
				# Write to SummaryWriter object only if it is validating
				if not is_test:
					# Consider average val loss of last 512 samples only after appending loss in each batch
					writer.add_scalar('val/loss', round(np.mean(losses[-(512//config.batch_size):]), 4), val_iter)
				# Prints for both validation and testing
				# Print only current batch loss (For ex: 0th, 512th, 1024th samples batches are printed)
				print(mode + ' Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				    epoch, int(batch_idx * len(data_place)), len(test_loader.dataset),
				    100. * batch_idx / len(test_loader), loss.item()))
	# Average Test Loss of current epoch
	test_loss = round(np.mean(losses), 4)
	return test_loss



def main(config, model):
	global test_boundary_prediction
	is_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if is_cuda else "cpu")
	print("Selected Torch Device: ", device)
	model = model.to(device)
	
	criterion = nn.CrossEntropyLoss()
	
	optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=5e-4)
	
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15])

	# If checkpoint directory is provided load it to the model and 
	# continue training from this checkpoint result
	if config.cp_dir:
		checkpoint = torch.load(config.cp_dir)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Prepare indexes to pass to CustomDataGenerator
	train_features_idxs, test_features_idxs = get_features_idxs(config)

	# Create a Dataset using CustomDataGenerator
	train_dataset = CustomDataGenerator(config, train_features_idxs)
	test_dataset = CustomDataGenerator(config, test_features_idxs)

	# Taking 80% samples of train_dataset for training and 20% samples for validation
	trainSize = int(len(train_features_idxs)*0.8)
	validSize = len(train_features_idxs)-(trainSize)
	
	trainSet, validSet = random_split(train_dataset, [trainSize, validSize], generator=torch.Generator().manual_seed(42))
	
	# Using DataLoader to pass to training, validation and testing
	train_loader = DataLoader(trainSet, batch_size = config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(validSet, batch_size = config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
	
	# Create a SummaryWriter object to visualize graphs using Tensorboard
	writer = SummaryWriter("runs/"+config.plots_dir, comment=f'LR_{config.learning_rate}_BS_{config.batch_size}')
	
	# Defining best val loss to infinite before starting training and validation
	best_val_loss = float('inf')
	
	for epoch in range(1, config.num_epochs+1):

		print('_____TRAINING_____')
		print(f'epoch #{epoch}')
		train_loss = train(config, model, device, train_loader, optimizer, scheduler, 
						   criterion, writer, epoch)
		print(f'\nEnd of Epoch #{epoch}, AVG Train Loss={train_loss}\n')

		print('_____VALIDATING_____')
		val_loss = test(config, model, device, val_loader, criterion, writer, epoch, is_test=False)
		print(f'\nEnd of Epoch #{epoch}, AVG Val Loss={val_loss}\n')
		# if current epoch val loss <= best val loss of previous epochs and 
		# if save_dir is provided then save the model as a checkpoint
		if val_loss <= best_val_loss and config.save_dir:
			best_val_loss = val_loss
			save_file_path = os.path.join(config.save_dir, 'model_{}_{:2.2f}.pth'.format(epoch, best_val_loss))
			states = {
						'epoch': epoch,
						'state_dict': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'best_val_loss': best_val_loss
					 }
			try:
				os.mkdir(config.save_dir)
			except:
				pass
			torch.save(states, save_file_path)
			print('Model saved ', str(save_file_path))

	print('_____TESTING_____')
	test_loss = test(config, model, device, test_loader, criterion, writer, epoch, is_test=True)
	print(f'\nAVG Test Loss={test_loss}\n')

	# ____EVALUATING____
	# Preparing ground truth dictionary and prediction dictionary for each movie in Test Set
	gt_dict = {}
	shot_end_frame_dict = {}
	for i in range(config.train_set_idx, len(config.movies_list)):
		with open(config.movies_list[i], 'rb') as f:
			cur_movie = pickle.load(f)
		cur_gt = []
		for item in cur_movie['scene_transition_boundary_ground_truth']:
			val = 1 if item else 0
			cur_gt.append(val) 
		gt_dict[config.movies_ref_dict[i]] = torch.tensor(cur_gt)
		shot_end_frame_dict[config.movies_ref_dict[i]] = cur_movie['shot_end_frame']
		assert len(test_boundary_prediction[config.movies_ref_dict[i]])==len(gt_dict[config.movies_ref_dict[i]])

	# Compute Average Precision (AP), Mean AVG Precision (mAP) and AP_dict for movie wise AP's
	AP, mAP, AP_dict = calc_ap(gt_dict, test_boundary_prediction)
	# Compute Mean MaxIoU (mean_miou) and miou_dict for movie wise miou's
	mean_miou, miou_dict = calc_miou(gt_dict, test_boundary_prediction, shot_end_frame_dict)

	# Close the SummaryWriter boject
	writer.flush()
	writer.close()
	print('Training Finished')
	print("\n____RESULTS____")
	print("Average Precision Metric \n", "\tAP =", AP, "\n\tmAP =", mAP, "\n\tAP_dict =", AP_dict)
	print("\nIoU Metric \n", "\tmean_miou =", mean_miou, "\n\tmiou_dict =", miou_dict)


if __name__ == '__main__':
	cnfg = Config(num_epochs=30, cp_dir='', save_dir='./checkpoints/')
	model = LGSS(cnfg)
	train = main(cnfg, model)