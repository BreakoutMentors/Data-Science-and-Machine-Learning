
from kaggle.api.kaggle_api_extended import KaggleApi
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

from get_kaggle_data import download_dataset
import os


def prepare_data(download_path, kaggle_link, kaggle_api):
    """
    This function downloads the dataset, and returns a HuggingFace Dataset with txt files.
    HuggingFace automatically groups the dataset by having each sample to a single line of text.
    The dataset contains the text with ['text'] key and ['attention_mask'] for the transformer.

    @params:
        download_path: The path to download the dataset
        kaggle_link: The link to the dataset on Kaggle
        kaggle_api: The Kaggle API to download the dataset

    @returns:
        Dataset with the text and attention_mask
    """

    # Getting current directory of where this file is running
    original_directory = os.getcwd()

    # Changing directory to the desired path, it will be created if does not exist
    try:
        os.chdir(download_path)
    except FileNotFoundError:
        os.mkdir(download_path)
        os.chdir(download_path)

    # Getting all the names of the files of the path 'download_path'
    directory_files = set(os.listdir())

    # Downloading the dataset
    download_dataset(kaggle_link, kaggle_api)
    # Getting name of folder that was unzipped
    folder_name = list(set(os.listdir()) - directory_files)[0]

    # Changing directory to current folder to read the txt files
    os.chdir(os.path.join(download_path, folder_name))

    # Getting all the names of the txt files
    file_names = os.listdir()

    # Loading the dataset with HuggingFace
    datasets = load_dataset('text', data_files={'train':file_names[0:350], 'valid':file_names[350:]})

    # Changing the current directory to the original directory where this file exists
    os.chdir(original_directory)
    return datasets

def tokenize_function(examples, tokenizer, block_size):
    """
    This function will take the text dataset and complete this steps below

    1. Tokenize the entire dataset
    2. Concatenate all examples from 2d list into a 1D
    3. Create blocks of the concatenated examples with a certain block size
    4. Create labels for the dataset

    @params:
        examples: The dataset to be tokenized
        tokenizer: The tokenizer to be used for tokenizing the dataset
        block_size: The size of the blocks to be created

    @returns:
        Tokenized dataset with labels
    """

    #1. Tokenize the entire dataset
    tokenized_examples = tokenizer(examples["text"])

    #2. Concatenate all examples from 2d list into a 1D
    # Going to flatten ['text'], ['input_ids'], ['attention_masks] from 2D lists to 1D lists or concatenate them
    concatenated_examples = {key:sum(tokenized_examples[key], []) for key in tokenized_examples.keys()}

    #3. Create blocks of the concatenated examples with a certain block size
    # Getting the total number of words
    num_tokens = len(concatenated_examples['input_ids'])
    # Getting the number of blocks; Cutting the that are left over that cannot make another block
    total_length = (num_tokens // block_size) * block_size

    results = {}
    for key, value in concatenated_examples.items():
        blocks = []
        for i in range(0, total_length, block_size):
            blocks.append(value[i: i+block_size])

        results[key] = blocks

    #4. Create labels for the dataset
    results['labels'] = results['input_ids'].copy()

    return results

def main():
    # Loading Kaggle Dataset link and Kaggle API to download dataset
    kaggle_link = "https://www.kaggle.com/mikhailgaerlan/spongebob-squarepants-completed-transcripts"
    api = KaggleApi()
    api.authenticate()


    model_name = 'distilgpt2'
    
    # Defining Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Adding padding token to the tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Getting datasets
    path = os.getcwd()
    raw_datasets = prepare_data(path, kaggle_link, api)

    # Tokenize datasets
    block_size = 128

    tokenized_datasets = raw_datasets.map(tokenize_function,
                                          batched=True, 
                                          batch_size=1000, 
                                          remove_columns=['text'], 
                                          fn_kwargs={'tokenizer':tokenizer, 'block_size':block_size})

    # Training Model
    training_args = TrainingArguments('train-test',
                                      num_train_epochs=10,
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch',
                                      learning_rate=5e-5,
                                      load_best_model_at_end=True
                                      )

    trainer = Trainer(model,
                      training_args,
                      train_dataset=tokenized_datasets['train'],
                      eval_dataset=tokenized_datasets['valid'])
    trainer.train()

    # Saving the model and uploading to HuggingFace repo
    repo_name = 'Breakout_Mentors_SpongeBob_Model'
    model.save_pretrained('model', push_to_hub=True, repo_name=repo_name)

    # This saves the tokenizer in the same directory
    tokenizer.save_pretrained('model')

    # Upload the tokenizer to the repo as well
    tokenizer.push_to_hub('Breakout_Mentors_SpongeBob_Model')
if __name__ == "__main__":
    main()