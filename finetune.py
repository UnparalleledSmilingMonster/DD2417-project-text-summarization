import os
import json
import requests
import pandas as pd
import numpy as np
import argparse
import torch
from tqdm.auto import tqdm
import subprocess


from ast import literal_eval

from datasets import Dataset,DatasetDict, Features, Sequence, Value, Array2D, load_dataset
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate






DATASET_ORIG_DIR = "AmazonFineFoods"
DATASET_DIR = "datasets"
DATASET_NAME = "dataset_tokenized.json"

MODEL_DIR = "models"
MODEL_DEFAULT_NAME = "gpt2-117M"
MODEL_NAME = "gpt2-117M-summary"
RATIO = {"train":0.9, "validation":0.05, "test":0.05}

OUTPUT_DEFAULT_DIR = os.path.join(MODEL_DIR, MODEL_DEFAULT_NAME)
OUTPUT_DIR = os.path.join(MODEL_DIR, MODEL_NAME)

NUM_SAMPLES = 50000

MAX_LENGTH = 512
MAX_LENGTH_LABEL = 128
MAX_LENGTH_GENERATION = 10
MIN_LENGTH_GENERATION = 2

OVERWRITE_OUTPUT_DIR = False
NUM_EPOCHS = 1
SAVE_STEPS = 100
BATCH_SIZE = 8
LEARNING_RATE = 5e-5



############################# Load the GPT-2 model ################################

def load_model():
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
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


#################### Format and trim Amazon Fine Food reviews ##############################

def process_data(data, tokenizer):
    print("Loading the original dataset and pre-processing it...")
    reviews = pd.read_csv(os.path.join(DATASET_ORIG_DIR, "Reviews.csv"), index_col=False)

    #only keep text and summary
    reviews = reviews.drop(
        ["Id", "ProductId", "UserId", "ProfileName","HelpfulnessNumerator", "HelpfulnessDenominator", "Score", "Time"],
        axis=1)           
    print("Number of reviews:", reviews.shape[0])

    #Clean the remaining dataset : we don't want an empty summary, or an empty text
    #Removes empty strings, but also strings only containing whitespaces
    #Removes text longer than 1024
    
    #Removing NaN values (strip replaces with None and does not remove)
    #print(reviews["Summary"][484367]) Example of nan in base dataset
    reviews.dropna(axis=0, how='any', inplace=True)

    reviews = reviews[reviews["Summary"].str.strip().astype(bool)]
    reviews = reviews[reviews["Text"].str.strip().astype(bool)]
    reviews["concat"] = reviews["Text"] + " TL;DR " + reviews["Summary"]
    reviews = reviews.sample(n=NUM_SAMPLES+ int(0.2*NUM_SAMPLES), random_state=127, axis=0)
    print("Number of clean reviews:", reviews.shape[0])
    reviews = reviews.drop(["Text", "Summary"], axis = 1)
    
    dataset = Dataset.from_pandas(reviews, split="train")
    dataset = dataset.map(lambda x : tokenizer(x["concat"], truncation=False, max_length=MAX_LENGTH), batched=True) #should not truncate since we filtered before 
    dataset = dataset.remove_columns(["concat","__index_level_0__"])
    
    df = dataset.to_pandas()
    df["length"] = df["input_ids"].apply(lambda x: len(x))
    print(df["length"].max())
    
    df = df[df["length"]< MAX_LENGTH]
    print("Right sized reviews :",df.shape[0])
        
    df = df[:NUM_SAMPLES]
    print("Final size of the dataset :",df.shape[0])    
    dataset = Dataset.from_pandas(df, split="train")     

    print(dataset.features)

    with open(data, "w") as f:
        for i in range(dataset.shape[0]):
            d = {"input_ids": dataset[i]["input_ids"],
                 "attention_mask": dataset[i]["attention_mask"]}
            f.write(json.dumps(d))
            f.write("\n")


#############################   Load the data and pre-process ###################################

def load_data(data):
    print("Loading the dataset...")
    dataset = load_dataset("json", data_files=data, split="train")
    print(dataset.features)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"]) #removed "labels"

    # 90% train, 20% test + validation
    train_test = dataset.train_test_split(test_size=RATIO["test"] + RATIO["validation"])
    # Split the 20% test + valid in half test, half valid
    test_valid = train_test['test'].train_test_split(test_size=RATIO["test"] / (RATIO["test"]+RATIO["validation"]))
    dataset = DatasetDict({
        'train': train_test['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})
    
    return dataset


#############################  Fine Tuning ##########################################

def pad_batcher(batch):
    """Batch is a list of dictionaries. Each dictionary is a row of the dataset.
    """
    batchsize = len(batch)
    inputs, masks = [], []
    for i in range(batchsize):
        inputs.append(
            torch.cat((batch[i]["input_ids"].reshape(1, -1),
                    torch.zeros(1, MAX_LENGTH - len(batch[i]["input_ids"]), dtype=torch.int32)), dim=1))
        masks.append(
            torch.cat((batch[i]["attention_mask"].reshape(1, -1),
                    torch.zeros(1, MAX_LENGTH - len(batch[i]["input_ids"]), dtype=torch.int32)), dim=1))
        """
        labels.append(
            torch.cat((batch[i]["labels"].reshape(1, -1),
                    torch.full((1, MAX_LENGTH - len(batch[i]["labels"])), -1, dtype=torch.int32)), dim=1))
        """
    return torch.cat(inputs, dim=0), torch.cat(masks, dim=0) #, torch.cat(labels, dim=0)


def finetune(model, tokenizer, train_dataset, eval_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=pad_batcher) #shuffle was set to False because cuda bug cpu
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=pad_batcher)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_batcher)
   
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)
    model.to(device)
    print(device)
    
    optimizer = AdamW(params = model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    # torch.set_default_device(device) #important

    loss_list, vloss_list = [], []
 
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(NUM_EPOCHS):
        enum  = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs, masks = batch
            inputs.to(device)
            masks.to(device)
            #labels.to(device)
            outputs = model(inputs.cuda(), attention_mask=masks.cuda(), labels = inputs.cuda())
            loss = outputs.loss
            loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            enum +=1
            if enum % 100 == 0 :
                print("Training loss:", loss)

        vloss = 0
        nbatches = 0
        model.eval()
        with torch.no_grad():
            for vbatch in eval_dataloader:
                vinputs, vmasks = vbatch
                vinputs.to(device)
                vmasks.to(device)
                voutputs = model(vinputs.cuda(), attention_mask=vmasks.cuda(), labels=vinputs.cuda())
                vloss += voutputs.loss
                nbatches += 1
            vloss /= nbatches
            vloss_list.append(vloss.item())
            print("Validation loss:", vloss)
        model.train()

        print("Model saved at epoch", epoch)
        model.save_pretrained(OUTPUT_DIR)
        """ #Code for removing old weights and replacing them after 1 epoch
        process = subprocess.run("rm -rv /gdrive/'My Drive'/project-DD2417/models/gpt2-117M-summary; cp -rv models/gpt2-117M-summary /gdrive/'My Drive'/project-DD2417/models; \
    rm -rv /gdrive/'My Drive'/ ", shell = True)
        """

    #For later plotting, etc.
    with open("losses.txt", "w") as f:
        for loss in loss_list:
            f.write(str(loss))
        f.write("\n")
        for loss in vloss_list:
            f.write(str(loss))
        f.write("\n")

    metric = evaluate.load('accuracy', 'rouge')  
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, masks = batch
            inputs.to(device)
            masks.to(device)
            outputs = model(inputs.cuda(), attention_mask=masks.cuda(), labels = inputs.cuda())

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["training"])

        metric.compute()
        print("ROUGE:", metric)

    """
    #Note : There is this class called Trainer that is provided by transformers and that you fill with all the right parameters. There are a lot. So I think it's better to use pytorch (see above) (or tensorflow) rather than this interface.
    
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

def eval_model(model, tokenizer, dataset):
    #rouge_metric= evaluate.load('rouge')
    rouge_metric = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    dataset["text"] = dataset["text"] + " TL;DR: "
    texts = dataset["text"].values.tolist()
    summaries = dataset["summary"].values.tolist()
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "left"
    
    #output = model.generate(**encoded_input,max_new_tokens=MAX_LENGTH_GENERATION, min_new_tokens =MIN_LENGTH_GENERATION,  do_sample=False,temperature = 1.0, top_k=50, num_beams = 2) #parameters to tune for the task
    
    rouge_1_prec =[]
    rouge_1_rec=[]
    
    rouge_2_prec =[]
    rouge_2_rec=[]
    
    rouge_L_prec =[]
    rouge_L_rec=[]
    for i in range(len(texts[:20])):
        encoded_input = tokenizer(text_target=texts[i], padding = True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        outputs = model.generate(**encoded_input,max_new_tokens=MAX_LENGTH_GENERATION, min_new_tokens =MIN_LENGTH_GENERATION,  do_sample=True, no_repeat_ngram_size=2,temperature =0.8, top_k=50, top_p=0.7)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens = True)
    
        prediction = prediction.split(' ')
        idx_tldr = prediction.index("TL;DR:")
        prediction = prediction[idx_tldr+1:]  
        prediction = ' '.join(prediction)   
                
        scores = rouge_metric.score(prediction, summaries[i])
        rouge_1_prec.append(scores["rouge1"].precision)
        rouge_1_rec.append(scores["rouge1"].recall)
        rouge_2_prec.append(scores["rouge2"].precision)
        rouge_2_rec.append(scores["rouge2"].recall)
        rouge_L_prec.append(scores["rougeL"].precision)
        rouge_L_rec.append(scores["rougeL"].recall)
#rouge_metric.compute()
        #print(dir(rouge_metric))
    
    print(max(rouge_1_prec))
    print(max(rouge_1_rec))


def summarize(model, tokenizer, text):
    """
    Given a text input, summarizes it.
    """
    text += " TL;DR: "
    encoded_input = tokenizer(text_target=text, truncation=True, max_length=MAX_LENGTH,return_tensors="pt")
    output = model.generate(**encoded_input,max_new_tokens=MAX_LENGTH_GENERATION, min_new_tokens =MIN_LENGTH_GENERATION,  do_sample=False,temperature = 1.0, top_k=50, num_beams = 2) #parameters to tune for the task
    
    #output = model.generate(**encoded_input,max_new_tokens=MAX_LENGTH_GENERATION, min_new_tokens =MIN_LENGTH_GENERATION,  do_sample=True, no_repeat_ngram_size=2,temperature =0.8, top_k=100, top_p=0.7)
    
    print("Summary :\n", tokenizer.decode(output[0], skip_special_tokens=True))


def main() :  
    def my_bool(s):
        return s != 'False'  
        
    default_text = "This product was very bad. It tasted bad and was so pricey for such little quality and quantity. I do not recommend it at all!"
        
   # Parse command line arguments
    parser = argparse.ArgumentParser(description='GPT-2 Parser')
    parser.add_argument('--finetune', '-ft', type=my_bool,  default=True, help='Whether to finetune or not.')
    parser.add_argument('--eval', '-e', type=my_bool, default = False, help = 'Whether to eval the model or not.')
    parser.add_argument('--input_sentence', '-i', type=str, default=default_text, help='The sentence to be summarized.')

    arguments = parser.parse_args()
    
    model, tokenizer = load_model()
    
    if arguments.finetune:
        data = os.path.join(DATASET_DIR, DATASET_NAME)
        if not os.path.isfile(data):
            process_data(data, tokenizer)
        dataset = load_data(data)
        print("Fine-tuning...")
        finetune(model, tokenizer, dataset["train"], dataset["validation"], dataset["test"])
    
    if arguments.eval :
        data_test = os.path.join(DATASET_DIR, DATASET_NAME_TEST)
        dataset_test = pd.read_json(data_test, lines = True)
        print("Evaluating the model...")
        eval_model(model, tokenizer, dataset_test)

    else:
        print("Summarizing the text inputted.")
        summarize(model, tokenizer, arguments.input_sentence)
    
    

if __name__ == '__main__' :
    main()   
    



