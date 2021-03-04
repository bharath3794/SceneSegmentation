# Scene Segmentation Using the MovieScenes Dataset (Solved as a Eluvio ML Challenge)

## Dataset
[Dataset Link by Eluvio](https://drive.google.com/file/d/1oZSOkd4lFmbY205VKQ9aPv1Hz3T_-N6e/view?usp=sharing)

This dataset is a modified version of MovieNet dataset. Original version can be found here: [MovieNet Original Dataset](https://github.com/movienet/movienet-tools)

## Train, Validation and Test Splits
Total Number of Movies in Dataset
>   64 Movies

#### Train Dataset:
>   * 90% of movies from these 64 Movies are created as Train Dataset (\~= 57 movies).
>   * All sample features created from these 57 movies are further split to Train and Val sets with 80% of total samples to Train and 20% of total samples to Validation Set
   
**NOTE**: Train and Validation sets are not splitted based on individual movies instead splitted on all the features created from every movie of these 57 movies (No. of features depends on no. of shots and sequence length we are considering for each feature)

#### Test Dataset:
>	10% of movies from these 64 Movies are created as Test Dataset (\~= 7 Movies)

**NOTE**: This test data is completely unseen and used only for testing purpose after all the epochs of training and validating

## Model
Taken from **AnyiRao's Model** available in the [Github Repo](https://github.com/AnyiRao/SceneSeg)
(All Credits to Anyi Rao. Thanks for the excellent work.)

### Architecture Diagram
![architecture](https://raw.githubusercontent.com/AnyiRao/SceneSeg/master/images/pipeline.png)
#### Ref:
>Rao, Anyi, et al. “A Local-to-Global Approach to Multi-modal Movie Scene Segmentation.”
arXiv.org, 2020, https://arxiv.org/abs/2004.02678

## Configuration Parameters
* Trained for 30 epochs with batch size of 8 (because of my GPU limitation)
* Learning Rate started with 1e-2

## Evaluation
### Loss
```
epoch #30
	AVG Train Loss = 0.2045
	AVG Val Loss = 0.2098
```

**On Test Set**
```
After epoch #30, 
        AVG Test Loss = 0.2833
```
**NOTE**: Testing is only performed once in the end after completing all the epochs of training and validation.

### Average Precision and IoU
#### Code Ref: 
**[Eluvio ML Challenges Github repo](https://github.com/eluv-io/elv-ml-challenge)**

**Average Precision**:
```	
        AP = 0.4243547650003294 
	mAP = 0.44133263058275096 
	AP_dict = {'tt1205489': 0.44937209408748796, 'tt1375666': 0.43944004051079305, 'tt1412386': 0.5537258781590395, 'tt1707386': 0.2981230102629725, 'tt2024544': 0.5690226130102841, 'tt2488496': 0.3108997275319894, 'tt2582846': 0.4687450505166901}
```
**IoU**:
```
	mean_miou = 0.053972835141407056 
	miou_dict = {'tt1205489': 0.028535499731287987, 'tt1375666': 0.07579754604309327, 'tt1412386': 0.05955595098601886, 'tt1707386': 0.030215081417225047, 'tt2024544': 0.030116858728477875, 'tt2488496': 0.08702184776057309, 'tt2582846': 0.06656706132317326}
```
**NOTE**: Average Precision and IoU are only computed for completely unseen Test set
#### Saved Checkpoint can be found here: [Best Checkpoint with Val Loss = 0.2029 at epoch #28](https://drive.google.com/file/d/14DBSDDr8rYyvyLEnDXUxRMXZC2VnuoQb/view?usp=sharing)
## Graphs
### Train Loss
![train](https://github.com/bharath3794/SceneSegmentation/blob/main/graphs_loss/train_loss_30_epochs.JPG)
### Val Loss
![val](https://github.com/bharath3794/SceneSegmentation/blob/main/graphs_loss/val_loss_30_epochs.JPG)

## Try Yourself
### Modules Required
```
numpy
torch
sklearn
tensorboard
glob
pickle
```
### Instructions to Run
- Set Configuration Parameters in `configuration.py` file or while initiating `Config` class object from `'__main__'` in `main.py` file

Important Parameters to Change:
```
data_folder: Change to your data path
save_dir: Give some path to save your checkpoints, otherwise checkpoints will not be saved
cp_dir: To resume training from certain checkpoint, give your checkpoint path here
num_epochs: number of epochs to run
batch_size: batch size to consider
n_seq: sequence length of each feature/sample
n_shots: no. of shots to consider in each sequence of a feature/sample
```
- Run `python main.py`
