## Segmenter for speech act classification

This folder contains the code for the segmenter used for preprocessing 
German parlamentary debates, to extract the input for speech act classification.


### Content of this repository:

```
- config 
	- segment.conf 	(the config file)
- data  
	- train.json 	(the train data)
	- dev.json		(the dev data)
	- test.json		(the test data)
- src	
	- segmenter.py 	(the segmentation script)
- README.md  		(this readme file) 
```


-------------------------
### How to run: 

#### Create anaconda virtual environment

```
conda create -n segmenter python=3.10
```

#### Activate virtual environment

```
conda activate segmenter
```

#### Install required packages
```
pip install torch torchvision
pip install transformers
pip install seqeval
pip install datasets
```

#### Train a model and test it
```
python src/segmenter.py config/segment.conf
```

The script creates the folder "models" where the trained 
model is stored.
Results are written to the "results" folder.

