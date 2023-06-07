# DD2417 Project : Generative Model for Text Summarization
- Timothée LY
- Troy FAU


## Requirements
The repository contains a copy of my conda virtual environment. Use the following command to copy it.
`conda env create -f environment.yml`

Then activate it : `conda activate gpt2`
To remove the environment:
`conda remove -n gpt2 --all`

To create the **requirements.txt** file from the current virtual env, use the command : 
`pip freeze > requirements.txt`

## Dataset 
Download **Amazon Fine Food Reviews** from the project topics file. Put the directory in the current directory.

## Fine-tuned model

The weights for GPT-2 are too heavy to be stored directly onto GitHub. Here is a [download link](https://drive.google.com/drive/folders/1PdwvIgehSPtqE1_8wv0ZzDrIfhZ0MK0D?usp=sharing) to the directory containing the fine-tuned model. Copy paste the directory **gpt2-117M-summary** in the **models/** directory of the repo.


## How to run 
To input the model a sentence to summarize, type in :  `python3 finetune.py -ft False -i "My text to summarize"`


A list of arguments is provided to the parser :
- -ft to finetune the model (default to True so actively set it to False in the arg parser)
- -e to eval the model
- -i to input a sentence 

You should only try the last one as the first one does not compile on cpu (we set the device to gpu as the cpu cannot handle the load and terminates the program)
and the second one will crash without the dataset properly generated.

    


