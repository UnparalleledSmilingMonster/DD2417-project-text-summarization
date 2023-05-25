import os
import requests
import pandas as pd
import argparse


from datasets import Dataset, load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


DATASET_ORIG_DIR = "AmazonFineFoods"
DATASET_DIR = "datasets"
DATASET_NAME = "dataset_amazon_tokenized.csv"

MODEL_DIR = "models"
MODEL_DEFAULT_NAME = "gpt2-117M"
MODEL_NAME = "gpt2-117M-summary"
RATIO = {"train":'[:80%]', "validation":'[80%:90%]', "test":'[90%:100%]'}

OUTPUT_DEFAULT_DIR = os.path.join(MODEL_DIR, MODEL_DEFAULT_NAME)
OUTPUT_DIR = os.path.join(MODEL_DIR, MODEL_NAME)

OVERWRITE_OUTPUT_DIR = False
NUM_TRAIN_EPOCH = 5
SAVE_STEPS = 100


############################# Load the GPT-2 model ################################

fine_tuned = os.path.join(MODEL_DIR, MODEL_NAME)
pre_trained = os.path.join(MODEL_DIR, MODEL_DEFAULT_NAME)

if not os.path.isdir(fine_tuned): #if no model trained yet
    print("Loading default GPT-2")
    if not os.path.isdir(pre_trained):
        model = GPT2LMHeadModel.from_pretrained("gpt2") #download if not yet saved locally
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(OUTPUT_DEFAULT_DIR)
        model.save_pretrained(OUTPUT_DEFAULT_DIR)                
    else :
        model = GPT2LMHeadModel.from_pretrained(pre_trained)
        tokenizer = GPT2Tokenizer.from_pretrained(pre_trained)
else : 
    print("Loading existing fine-tuned model")
    model = GPT2LMHeadModel.from_pretrained(fine_tuned)
    tokenizer = GPT2Tokenizer.from_pretrained(pre_trained) #For now no change to the tokenizer       
 
#################### Format and trim Amazon Fine Food reviews ##############################

data = os.path.join(DATASET_DIR, DATASET_NAME)

if not os.path.isfile(data):
    print("Loading the original dataset and pre-processing it.")
    reviews = pd.read_csv(os.path.join(DATASET_ORIG_DIR, "Reviews.csv"), index_col=False)
    #print(reviews.head())

    #only keep text and summary
    reviews = reviews.drop(["Id", "ProductId", "UserId", "ProfileName","HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time"], axis=1)           
    print(reviews.head())
    print("Number of reviews :", reviews.shape[0])

    #Clean the remaining dataset : we don't want an empty summary, or an empty text
    #Removes empty strings, but also strings only containing whitespaces
    #Removes text longer than 1024
    reviews = reviews[reviews["Summary"].str.strip().astype(bool)]
    reviews = reviews[reviews["Text"].str.strip().astype(bool)]
    
    #reviews = reviews[reviews["Text"].str.split().str.len().lt(1024)]   #does not seem to work well, handled in tokenizer
    #reviews = reviews[reviews["Summary"].str.split().str.len().lt(1024)]  
    

    print("Number of clean reviews :", reviews.shape[0])
    dataset =Dataset.from_pandas(reviews, split='train') #no split for now
    print(dataset.features)    
    dataset = dataset.map(lambda x : tokenizer(x["Text"], truncation = True, max_length = 1024), batched=True)
    dataset.to_csv(data, index = None) 
   
#############################   Load the data and pre-process ###################################

print("Loading the dataset")

dataset = load_dataset("csv", data_files=data, split = "train")
dataset.set_format(type="torch", columns=["Text", "input_ids", "attention_mask", "Summary"])
dataset.format['type']

print(dataset.features)

split = ['train'+RATIO["train"], 'validation'+RATIO["validation"], 'test'+RATIO["test"]]

#dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "summary"])

#data_collator = load_data_collator(tokenizer, mlm = False)

#############################  Fine Tuning ##########################################

def finetune(model = model, tokenizer = tokenizer, train_dataset = train_dataset, eval_dataset = eval_dataset, test_dataset = test_dataset, data_collator = data_collator):

    """
    #Note : There is this class called Trainer that is provided by transformers and that you fill with all the right parameters. There are a lot. So I think it's better to use pytorch (or tensorflow) rather than this interface.
    
    training_args = TrainingArguments(
          output_dir=MODEL_DIR,
          overwrite_output_dir=True,
          num_train_epochs=NUM_TRAIN_EPOCHS,
          saves_steps = SAVE_STEPS
      )
      
    trainer = Trainer(
          model=model,
          tokenizer=tokenizer,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          test_dataset=test_dataset
    )
    trainer.train()
    trainer.eval()
    trainer.save_model()
"""

    
 
def summarize():
    """
    Given a text input, summarizes it.
    """
    return ""
    
    
 
def main() :    
   # Parse command line arguments
    parser = argparse.ArgumentParser(description='GPT-2 Parser')
    parser.add_argument('--finetune', '-ft', type=bool,  default = True, help='Whether to finetune or not.')
    parser.add_argument('--input_sentence', '-i', type=str, required=True, help='The sentence to be summarized.')

    arguments = parser.parse_args()
    
    if arguments.finetune :
        print("Starting the fine-tuning")
        finetune()
    else :
        summarize(arguments.input_sentence)
    
    

if __name__ == '__main__' :
    main()   
    



