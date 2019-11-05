## Setup

You are going to work on the Notebook VM you created [earlier](../1-new-workspace/1-setup-compute.md). 

1. To get started, first navigate to the JupyterLab instance running on the Notebook VM by clicking on the JupyterLab link shown below:
![](log_in.png)

1. After going through authentication, you will see the JupyterLab frontend. As you authenticate, make sure to use the same user to log in as was used to create the Notebook VM, or else your access will be denied. Next open an Terminal (either by File/New/Terminal, or by just clicking on Terminal in the Launcher Window).
![](terminal.png)

1. In the terminal window clone this repository by typing:

        git clone https://github.com/danielsc/azureml-workshop-2019

2. After the clone completes, in the file explorer on the left, navigate to the folder `2-interpretability` and open the notebook `1-simple-feature-transformations-explain-local.ipynb`:
![](notebook.png)

Now follow the instructions in the notebook.

## VS Code Extensions used

Visual Studio Code Remote Development Extension Pack:
https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack#review-details

Azure Machine Learning for Visual Studio Code:
https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai

