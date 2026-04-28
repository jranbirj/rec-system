Hello! We're excited for you to run out project, and there are defintely some specific run details that you need to know before you do! 

There are five .ipynb files that we have run and setup for google colab. Other than that there is the output files of those runs, and there is also the 
Gradio folder for the Vector DataBase and those files that you have to handle. 

This outline will be written in the steps that we suggest (and is necessary) to run each file

## Important 
## Excluding errors and bugs (which we discuss later) the only edits that you should have to make to our colab files are the paths to import directories for reading the datasets and output directories for the embeddings, vectordatabase, images, and more. 

Running 01. 
To run 01 on google colab, you can load our file, ensure that the data from the Yelp Open Dataset is in a folder on your drive, and ensure that the path statement is right

Running 02. 
1) Change path to your drive yelp_academic_dataset_business.json
2) change path to your drive yelp_academic_dataset_review.json
3) change path to your drive yelp_academic_dataset_user.json
4) change path to your desired save_dir (save directory) this will be where 03, 04, and 05 all pull from. 

Running 03
1) Change path of origin_dir to 02's save_dir in google drive, and change the name of the desired resuls folder. 
2) We have set a seed to reproduce our work at 42 across all future runs 

## We ran into errors within our kernel where numpy had to be reinstalled, usually after one install this fixed it. 


## For 04 and 05 we ran into issues with the PyTorch installation, this should no longer be an issue from the pip install at the top of both files, but it is something to keep in mind. 
Running 04. 
1) Change data_dir path 
2) change results_dir path 
3) change export_dir path (for qdrant vector database files)
4) change results_dir path at bottom of notebook for comparison vs SVD baseline to the save_dir of 03 notebook. This is not necessary but useful if want a one to one comparison right away 

Running 05. 
1) change data_dir path to output of 02
2) change results_dir, ckpt_dir, config_dir, history_dir, and plots_dir to desired outputs