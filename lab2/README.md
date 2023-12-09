# Lab 2

## Table of Contents
1. [Quick walkthorugh](#quick-walkthorugh)
2. [File structure](#file-structure)
3. [How to run](#how-to-run)
4. [How to improve results](#how-to-improve-the-wer-result)  
    a. [Model-centri approach](#model-centric-approach)  
    b. [Data-centric approach](#data-centric-approach)  
5. [Link to models](#models)

## Quick walkthorugh
This project aims at fine-tuning a whisper model in my mothertounge for ASR tasks.
It then aims at showcaseing this model in a gradio application.

The gradio application can be found and tested here: [link](https://huggingface.co/spaces/AdrianHR/Whisper-Transcribe-Swedish-Songs)

If you input the name of a swedish artist and a song title of theirs, you should get  
four things:
1) A quick soundbite from that song, where the vocals have been "separated"
2) Lyrics for that song from genius
3) Lyrics that have been transcribed by the model
4) WER between actual lyrics and transcribed lyrics

Note: The model is finetuned on speech data and not songs, and background singing and  
sometimes instruments do make the transcribed lyrics really bad sometimes. But it is  
quite fun given the size of the project.

The repo where the gradio application was written does have most of the commit history  
so I would recommend looking there if you want to see the history it can be found [here](https://huggingface.co/spaces/AdrianHR/Whisper-Transcribe-Swedish-Songs/tree/main)

The feature-pipeline and the training-pipeline were run in Colab. I would recommend to  
do so yourself if you are thinking of running this code. Those files are available here:  
* [feature-pipeline](https://colab.research.google.com/drive/1eF0fgZC9dvvXmMbnrjpxTQDjqMJ488Cv?usp=sharing)
* [training-pipeline](https://colab.research.google.com/drive/1sr2ZoyBzas4k13hEFBSLfLP3AAJ8ENmA?usp=sharing)

Also it should be said that this project is heavily inspired by the Colab Notebook by  
Sanchit Gandhi for fine-tuning the model, checkpointing it and placing it on hf.
That notebook is available [here](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb)

## File structure
```bash
├── Feature_Pipeline_Whisper.ipynb      <- Here we load the data from hf and turn preprocess and save it to drive
├── Training_Pipeline_Whisper.ipynb     <- Fine-tune whisper on the data. Save checkpoints to drive and huggingface.
├── README.md                           <- Describe the contents and goal of the project and how to run it.
├── example.png                         <- Example of why it might be a good idea to transcribe on the foreground audio
└── Whisper-Transcribe-Swedish-Songs    <- Detailing the HuggingFace Gradio application
    ├── app.py                          <- Runs the application
    ├── audio.wav                       <- Example Audio file
    ├── requirements.txt                <- Requirements to run the gradio application
    └── README.md                       <- Auto-Generated README from huggingface space.
```

## How to run
To run this project it is not necessary to clone the project.

Make copies of the pipelines for handling features and model training
* [feature-pipeline](https://colab.research.google.com/drive/1eF0fgZC9dvvXmMbnrjpxTQDjqMJ488Cv?usp=sharing)
* [training-pipeline](https://colab.research.google.com/drive/1sr2ZoyBzas4k13hEFBSLfLP3AAJ)

Run it on Colab and by logging into your on huggingface account, store the models there.

Then, create a free space on huggingface and fill that repo with the files in the  
[Whisper-Transcribe-Swedish-Songs](lab2/hf-space_Whisper-Transcribe-Swedish-Songs) directory.

If you still wish to run it locally, it will still be similar, but you will have to  
refactor the code a bit. If you have any questions on that please contact me.

## How to improve the WER result?

Well there are two main ways to improve the general WER of the models.
Either by data augmentation, training with more or better data or by  
improving the model itself, finding better hyperparameters and such.

### Model Centric Approach
This could be either by 
* Tuning the hyperparameters more specifically it could be by:
    * Letting the model learn for longer to fit better to the data. I.e. more steps
    * Instead avoiding overfitting by not training for too long. I.e. less steps or better constraints.
    * Changing the learning rate could be good, if it is too big we might miss a better solution  
    but if it is too low it could become really slow.
    * We could increase the batch-size in order to take more training examples into account  
    at each step of the training, before changing the weights of the model.
* Change the type of model:
    * Fine-tune another whisper model, this could either be a larger model such as medium or large-v2  
    But it could also be a multi-lingual one as opposed to only English.
    * Chose a less complex model such as Wav2Vec

### Data-Centric Approach
To this could be either a later version of the used common_voice dataset such as [common_voice 13.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/viewer/sv-SE)  
or maybe a completely other dataset such as this  one that was uploaded by KTH [NST Swedish ASR Database](https://huggingface.co/datasets/KTH/nst)


## Models
Here are two of the models that i did save on hf:
* [model-v1](https://huggingface.co/AdrianHR/whisper-small-sv)
* [model-v4](https://huggingface.co/AdrianHR/whisper-small-sv-v4)

They do come with a history of their performance during training among other things.