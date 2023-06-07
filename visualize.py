import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

def histogram_data(data, bins = 20):
    n, bins, patches = plt.hist(data, bins, facecolor='blue', alpha=0.7)
    plt.yscale('log')
    plt.xlabel("Number of words in the documents")
    plt.ylabel("Number of documents")

    plt.show()
    

"""

df=pd.read_csv("lengths.csv", index_col=False)
print(df.head)

texts = df["text_length"].values.tolist()
summaries = df["summary_length"].values.tolist()


histogram_data(summaries)
histogram_data(texts, bins = 100)
"""

def plot_val_loss(val):
    plt.xlabel("Epoch #",fontsize=14)
    plt.ylabel("Loss",fontsize=14)
    x = np.arange(len(val))
    plt.plot(x,val)
    plt.title("Validation loss per epoch",fontsize=16)
    plt.show()
    
def plot_train_loss(train):    
    plt.xlabel("Batch #",fontsize=14)
    plt.ylabel("Loss",fontsize=14)
    x=np.arange(len(train))
    plt.plot(x,train)
    plt.title("Training loss per batch(16)",fontsize=16)
    plt.show()

def plot_losses(file):
    training_losses = []
    validation_losses = []

    with open(file, 'r') as file:
        lines = file.readlines()        
        training_loss = literal_eval(lines[0].split(":")[1][1:])
        validation_loss = literal_eval(lines[1].split(":")[1][1:])
        print(len(validation_loss))
        plot_val_loss(validation_loss)
        plot_train_loss(training_loss)
        

    
        
plot_losses("losses_clean.txt")
