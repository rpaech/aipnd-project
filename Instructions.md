# Project instructions

## Part 1: Developing an Image Classifier with Deep Learning

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. We'll provide some tips and guide you, but for the most part the code is left up to you. As you work through this project, please refer to the rubric for guidance towards a successful submission.

Remember that your code should be your own, please do not plagiarize (see here for more information).

This notebook will be required as part of the project submission. After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this project. If you'd instead prefer to work on your local machine, you can find the files on GitHub here.

If you are using the workspace, be aware that saving large files can create issues with backing up your work. You'll be saving a model checkpoint in Part 1 of this project which can be multiple GBs in size if you use a large classifier network. Dense networks can get large very fast since you are creating N x M weight matrices for each new layer. In general, it's better to avoid wide layers and instead use more hidden layers, this will save a lot of space. Keep an eye on the size of the checkpoint you create. You can open a terminal and enter ls -lh to see the sizes of the files. If your checkpoint is greater than 1 GB, reduce the size of your classifier network and re-save the checkpoint.

### Instructions regarding using the Classroom Workspace

As you work in the classroom workspace, please do not to leave extremely large .pth files inside the /home/workspace directory. When you run your code, you need to delete those .pth files before moving to a different tab or closing your browser tab.

Failing to do so will cause the workspace to keep on loading and you will encounter the following screen.

## Part 2: Building the command line application

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.
Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.

### 1. Train

Train a new network on a data set with train.py

- Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
  - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
  - Choose architecture: python train.py data_dir --arch "vgg13"
  - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
  - Use GPU for training: python train.py data_dir --gpu

### 2. Predict

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

- Basic usage: python predict.py /path/to/image checkpoint
- Options:
  - Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
  - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
  - Use GPU for inference: python predict.py input checkpoint --gpu

The best way to get the command line input into the scripts is with the argparse module in the standard library. You can also find a nice tutorial for argparse here.

### Caution - Compute and Storage Capacity of Workspaces

Each workspace is provided with adequate compute capacity and storage space (~2.5Gb) to accommodate the ML models. Sometimes, when you return to your workspace, you might face difficulty restoring the workspace because of the presence of bulky interim files generated during training ML models. So, avoid filling up your workspace by doing a simple step mentioned below.

#### a. How to prevent filling up your workspace?

As you go on to train your models, the models will be saved in your /home/workspace/saved_models directory along with the path configuration (*.pth) files created in the project directory. Both these "saved_models" directory and ".pth" files are bulky interim files, meaning they will be created every time you attempt to train your ML model. These files with a total size > 2.5Gb can fill up your workspace to the brim, thus causing the workspace-restoration problems when you'd return to the workspace next time.

Therefore, we strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.

#### b. Where do I save the "heavy" files having total size > 2.5Gb, if required?

You can move your heavy files, such as *.pth file, into the ~/opt directory, where you will have tons of space. Your files will be available there as long as you are working in a given session. Sessions are terminated after 30 minutes of inactivity. All your files in the ~/opt are temporary, and will not restore in the next session.

> Note: For the current exercise, you don't need Gigabytes of data, and in general, students should never submit projects with that much data either.

#### c. What if you neither delete these heavy files nor move them to the ~/opt directory before closing your workspace session?

In such a case, you might face a workspace-restoration problem, and the only way out is to contact Udacity support.

## Project Rubric

### Files Submitted

- Submission Files:  The submission includes all required files. (Model checkpoints not required.)

### Part 1 - Development Notebook

- Package Imports:  All the necessary packages and modules are imported in the first cell of the notebook
- Training data augmentation:  torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
- Data normalization:  The training, validation, and testing data is appropriately cropped and normalized
- Data batching:  The data for each set is loaded with torchvision's DataLoader
- Data loading:  The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
- Pretrained Network:  A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
- Feedforward Classifier:  A new feedforward network is defined for use as a classifier using the features as input
- Training the network:  The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
- Testing Accuracy:  The network's accuracy is measured on the test data
- Validation Loss and Accuracy:  During training, the validation loss and accuracy are displayed
- Loading checkpoints:  There is a function that successfully loads a checkpoint and rebuilds the model
- Saving the model:  The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary
- Image Processing:  The process_image function successfully converts a PIL image into an object that can be used as input to a trained model
- Class Prediction:  The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image
- Sanity Checking with matplotlib:  A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

### Part 2 - Command Line Application

- Training a network:  train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint
- Training validation log:  The training loss, validation loss, and validation accuracy are printed out as a network trains
- Model architecture:  The training script allows users to choose from at least two different architectures available from torchvision.models
- Model hyperparameters:  The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
- Training with GPU:  The training script allows users to choose training the model on a GPU
- Predicting classes:  The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
- Top K classes:  The predict.py script allows users to print out the top K classes along with associated probabilities
- Displaying class names:  The predict.py script allows users to load a JSON file that maps the class values to other category names
- Predicting with GPU:  The predict.py script allows users to use the GPU to calculate the predictions
