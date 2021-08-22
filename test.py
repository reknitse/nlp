#!/usr/bin/env python

from transformers import pipeline
from datasets import load_dataset

############################################################################################################
# CONFIG
############################################################################################################
# candidates for the categories
candidate_categories = ["Verbrechen", "Tragödie", "Stehlen"]
# path of the csv file (either absolute or relative to the root directory from where the script is running
csv_file = "test_data.csv"
# delimiter used in the csv file
delimiter = "|"
# column name of the text to be analysed
text_column_name = "text"
# optional name of the column with the correct category
category_column_name = "category"
############################################################################################################
classifier = pipeline("zero-shot-classification",
                      model="Sahajtomar/German_Zeroshot")

datasets = load_dataset("csv", data_files={"test": csv_file}, delimiter=delimiter, download_mode="force_redownload")

print("-------------------------------------")
print("Starting analysis")

for dataset in datasets["test"]:
    sequence = dataset[text_column_name]
    print("Analysing text >>" + sequence + "<<")
    output = classifier(sequence, candidate_categories)
    category = output["labels"][0]
    print("This text is {:s} ({:.2f}%)".format(category, output["scores"][0]*100))
    if category_column_name in dataset.keys() and dataset[category_column_name]:
        print("It should be " + dataset[category_column_name])
        if category == dataset[category_column_name]:
            print("-------> CORRECT")
        else:
            print("-------> WRONG")
