from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import argparse
import os
from os.path import isfile, join
import pandas as pd

"""
Get list of models we want to use. 

Model directories should contain `config.json` and `pytorch_model.bin`.
"""
def get_models(args):
    if args.model_dir:
        models_list = []
        for entry_name in os.listdir(args.model_dir):
            entry_path = os.path.join(args.model_dir, entry_name)
            if os.path.isdir(entry_path):
                models_list.append(entry_name)
        return models_list
    elif args.model_path:
        return [args.model_path]
    else:
        return False

"""
Get list of dataframes we want to process. 

Dataframes should contain `sentence` column (ie. column to obtain sentiment for).
"""
def get_input_dfs(args):
    if args.input_dir:
        sentences_df_list = [f for f in os.listdir(args.input_dir) if isfile(join(args.input_dir, f)) and f.endswith('.csv')]
        return sentences_df_list
    elif args.text_path:
        return [args.text_path]
    else:
        return False

"""
Create output directory for processed dataframe (ie. [output_dir]/[model_name]) 
"""
def make_output_directory(output_dir, models_list):
    for model in models_list:
        if not os.path.exists(os.path.join(output_dir, model)):
            os.mkdir(os.path.join(output_dir, model))

parser = argparse.ArgumentParser(description='Sentiment analyzer')

parser.add_argument('-a', action="store_true", default=False)

parser.add_argument('--text_path', type=str, help='Path to the text file.')
parser.add_argument('--output_dir', type=str, help='Where to write the results')
parser.add_argument('--model_path', type=str, help='Path to classifier model')
parser.add_argument('--input_dir', type=str, help='Path to multiple text files')
parser.add_argument('--model_dir', type=str, help='Path to multiple models')

args = parser.parse_args()

sentences_df_list = get_input_dfs(args)
print('Getting list of .csv files to process...')
print(sentences_df_list)
print('Getting list of model directories...')
models_list = get_models(args)
print(models_list)
make_output_directory(args.output_dir, get_models(args))

for model_path in models_list:
    model_dir = os.path.join(args.model_dir, model_path)
    print('Loading {} model....'.format(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=3,cache_dir=None)
    print('{} model loaded. Processing .csv files...'.format(model_path))
    output_dir = os.path.join(args.output_dir, model_path)
    for df_filename in sentences_df_list:
        print('Processing {} using {} model...'.format(df_filename, model_path))
        df = pd.read_csv(os.path.join(args.input_dir, df_filename))
        text_inputs = df['sentence'].tolist()
        final_df = predict(text_inputs,model,write_to_csv=False)
        processed_csv_filename = '{}_{}'.format(model_path, df_filename)
        print('Finished processing {}. Saving as .csv file named {}...'.format(df_filename, processed_csv_filename))
        final_df.to_csv(os.path.join(output_dir, processed_csv_filename), index=False)
        
# if not os.path.exists(args.output_dir):
#     os.mkdir(args.output_dir)

# if args.text_path.endswith('.txt'):
#     with open(args.text_path,'r') as f:
#         text = f.read()
#         text_inputs = text.strip().split('\n\n')
# elif args.text_path.endswith('.csv'):
#     df = pd.read_csv(args.text_path)
#     text_inputs = df['sentence'].tolist()

# model = AutoModelForSequenceClassification.from_pretrained(args.model_path,num_labels=3,cache_dir=None)

# output = "75_predictions.csv"
# final_df = pd.DataFrame()
# final_df = predict(text_inputs,model,write_to_csv=False)
# final_df.to_csv(os.path.join(args.output_dir,output), index=False)

# for text in text_inputs:
#     print(text)
#     df_output = predict(text,model,write_to_csv=False)
#     final_df = pd.concat([final_df, df_output])

# predict(text,model,write_to_csv=True,path=os.path.join(args.output_dir,output))