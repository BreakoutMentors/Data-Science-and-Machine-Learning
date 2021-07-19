from transformers import pipeline
from tokenizers import Tokenizer
import streamlit as st

# Function to clean text-output
def clean_output(text):
    """
    This takes the text-generated output of the model and cleans it up for printing purposes
    @param text: The text-generated output of the model
    @return: The cleaned up text-generated output
    """
    punctuations = ['!', '.', '?', ']']
    text_split = text.split(':')

    for i in range(len(text_split)):
        quote = text_split[i]
        last_idxs = []
        for punctuation in punctuations:
            try:
                last_idxs.append(quote.rindex(punctuation))
            except ValueError:
                continue

        if len(last_idxs) > 0:
            if i == len(text_split)-1:
                last_puncation = max(last_idxs)
                quote = quote[:last_puncation+1]
            else:
                last_puncation = max(last_idxs)
                # This checks if Mr. Krabs is the last word in the quote
                if quote[last_puncation-2:last_puncation+1] == 'Mr.':
                    quote = quote[:last_puncation-2] + '\n' + quote[last_puncation-2:]
                else:
                    quote = quote[:last_puncation+1] + '\n' + quote[last_puncation+1:]
        text_split[i] = quote

    return ':'.join(text_split)

@st.cache(hash_funcs={Tokenizer:id})
def get_pipeline(task, repo_path):
    return pipeline(task, model=repo_path)

# Loading model using Pipeline to easily use the model
task = 'text-generation'
repo_path = "Coldestadam/Breakout_Mentors_SpongeBob_Model"
generator = get_pipeline(task, repo_path)

# Giving a title to the app
st.title("SpongeBob Script-Generator")

# Showing GIF
st.image("https://media2.giphy.com/media/nDSlfqf0gn5g4/giphy.gif")

# Explanation
st.subheader("How does this app work?")

st.markdown("""This app uses [HuggingFace](https://huggingface.co/), an open repository and library to pretrained transformer models. 
A [smaller version of GPT-2](https://huggingface.co/distilgpt2) that was pretrained was finetuend on scripts of the show [SpongeBob SquarePants](https://www.kaggle.com/mikhailgaerlan/spongebob-squarepants-completed-transcripts), then an app was implemented and hosted with [Streamlit](https://streamlit.io/).""")
st.subheader("What are tokens?")
st.write("Tokens are split up pieces of a string, such as words.")
st.code("""sentence='Hi my name is Adam!'
tokenized_sentence = tokenizer(sentence)
print(tokeized_sentence)
>> ['Hi', 'my', 'name', 'is', 'Adam', '!']""")
st.write("You can see that sentence is broken down to smaller components such as words separated by spaces, which is an example of subword tokenization.")
st.markdown(
"""There are other types of tokenization techniques used in Natural Language Processing, including Byte-Pair Encoding (BPE) that was used for GPT2.

Here are two sources to read more:

1. [HuggingFace Summary](https://huggingface.co/transformers/tokenizer_summary.html#summary-of-the-tokenizers)
2. [FloydHub Blog](https://blog.floydhub.com/tokenization-nlp/)""")

st.write("The takeaway here is that the more tokens you want to generate, the greater number of words you get in return.")

st.subheader("Github Repository")
st.markdown("Please look at the code to see how everything was completed [here](https://github.com/Coldestadam/SpongeBob-Text-Generation).")

st.header("Generate Script Below:")

# Getting the number of tokens for the model with a text box
num_tokens = st.number_input('Number of Tokens to generate:', min_value=100, max_value=1000, step=1)

# Using a select box to choose the first character for the script
characters = ['SpongeBob', 'Patrick', 'Squidward', 'Gary', 'Mr. Krabs']
selected_character = st.selectbox('Select the character to begin the script', characters)

# Generating the model and wiritng it after the 'Generate Script' button is pressed
if st.button('Generate Script'):
    # Getting the text-generated output of the model
    output = generator(selected_character+":", max_length=num_tokens, min_length=num_tokens, do_sample=True, top_k=50)
    # Cleaning up the text-generated output
    cleaned_output = clean_output(output[0]['generated_text'])
    # Writing the text-generated output to the app one line at a time
    for quote in cleaned_output.split('\n'):
        st.write(quote)