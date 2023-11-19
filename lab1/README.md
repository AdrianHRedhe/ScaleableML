# Lab 1 - Wine

This lab is aimed at making a try at solving a classic ML problem, wine quality classification.  
The main focus of the lab is however to try to build a serverless architecture to do so in a distributed way.  

The functions are run via hopsworks and two web applications are hosted on huggingface that showcase  
2 different things:
1. Is a monitor application that can be used to look at the history of the runs of the inference pipeline.  
It also keeps track of model performance over time.  
It can be found here [https://huggingface.co/spaces/AdrianHR/wine-monitor](https://huggingface.co/spaces/AdrianHR/wine-monitor)


2. Is a *simulation* application where a user can put in arbitrary numbers for the feature vectors.  
It allows users to create *fake* wines and predict their classes.  
It can be found here [https://huggingface.co/spaces/AdrianHR/wine](https://huggingface.co/spaces/AdrianHR/wine)

## How to run
The dependencies are quite tricky with this lab, given that it needs a very specific version of gradio and hopsworks  

I would recommend you to create a new conda environment to run this.

**To run it**

Clone the repository

```bash
  git clone https://github.com/AdrianHRedhe/ScaleableML.git
```

Go to the lab directory

```bash
  cd ScaleableML/lab1/wine
```

Create a new conda env

```bash
  conda create -n sml-lab1
```

Install python 3.10 specifically

```bash
  conda install python==3.10
```

Install all the requirements

```bash
  pip install -r requirements.txt
```

Then to run I would do things in the following order:

* notebook EDA & Backfill. Can be found [here](wine/wine-eda-and-backfill-feature-group.ipynb)
* notebook training pipeline. Can be found [here](wine/wine-training-pipeline.ipynb)
* feature pipeline. Can be found [here](wine/wine-feature-pipeline-daily.py)
* batch inference. Can be found [here](wine/wine-batch-inference-pipeline.py)

You will need a hopsworks API key to run the project. You can create a free account on their website.
Then you can just put the API key into github actions, then you can just run the program.

