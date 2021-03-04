# Scene Segmentation Using the MovieScenes Dataset

## Dataset
https://drive.google.com/file/d/1oZSOkd4lFmbY205VKQ9aPv1Hz3T_-N6e/view?usp=sharing
This dataset is a subset of MovieNet dataset which can be found here http://movienet.site/

## Train, Validation and Test Splits
Total Movies in Dataset:
   64 Movies
Train Dataset:
   90% of movies from these 64 Movies are created as Train Dataset (\~= 57 movies)
   All sample features created from these 57 movies are further split to Train and Val sets with 80% of total samples to Train and 20% of total samples to Val Set
   **_NOTE_**: Train and Val sets are not splitted based on individual movies instead splitted on all the features created from every movie of these 57 movies (No. of features depends on no. of shots sequence length we are considering for each feature)
Test Dataset:
	10% of movies from these 64 Movies are created as Test Dataset (\~= 7 Movies)
	**_NOTE_**: This data is completely unseen and used only for testing purpose after all epochs of training and validating

## Model
Taken from AnyiRao's Model available in his Github Repo (https://github.com/AnyiRao/SceneSeg)
Thanks to his excellent work.
**Architecture Diagram**
https://raw.githubusercontent.com/AnyiRao/SceneSeg/master/images/pipeline.png
Ref:
Rao, Anyi, et al. “A Local-to-Global Approach to Multi-modal Movie Scene Segmentation.”
arXiv.org, 2020, https://arxiv.org/abs/2004.02678

## Configuration Parameters
Trained for 30 epochs with batch size of 8 (because of my GPU limitation)
Learning Rate started with 1e-2

## Evaluation
**LOSS**
epoch #30
	AVG Train Loss = 0.2045
	AVG Val Loss = 0.2098

**On Test Set**
After epoch #30, AVG Test Loss = 0.2833
**_NOTE_**: Testing is only performed once in the end after completing all the epochs of training and validation.

**Average Precision and IoU**
Code Ref: Eluvio ML Challenges Github repo (https://github.com/eluv-io/elv-ml-challenge)

**_Average Precision_**:
	AP = 0.4243547650003294 
	mAP = 0.44133263058275096 
	AP_dict = {'tt1205489': 0.44937209408748796, 'tt1375666': 0.43944004051079305, 'tt1412386': 0.5537258781590395, 'tt1707386': 0.2981230102629725, 'tt2024544': 0.5690226130102841, 'tt2488496': 0.3108997275319894, 'tt2582846': 0.4687450505166901}

**_IoU_**:
	mean_miou = 0.053972835141407056 
	miou_dict = {'tt1205489': 0.028535499731287987, 'tt1375666': 0.07579754604309327, 'tt1412386': 0.05955595098601886, 'tt1707386': 0.030215081417225047, 'tt2024544': 0.030116858728477875, 'tt2488496': 0.08702184776057309, 'tt2582846': 0.06656706132317326}

**_NOTE_**: Average Precision and IoU are only computed for completely unseen Test set

## Graphs
**Train Loss**

**Val Loss**


