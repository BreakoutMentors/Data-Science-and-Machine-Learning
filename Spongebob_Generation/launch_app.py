from transformers import pipeline
import streamlit as st

# Loading model using Pipeline to easily use the model
repo_path = "Coldestadam/Breakout_Mentors_SpongeBob_Model"
generator = pipeline('text-generation', model=repo_path)

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
                quote = quote[:last_puncation+1] + '\n' + quote[last_puncation+1:]
        text_split[i] = quote

    return ':'.join(text_split)

# Giving a title to the app
st.title("SpongeBob NLP App")

# Getting the number of tokens for the model with a text box
num_tokens = st.number_input('Number of Tokens:', min_value=20, max_value=10000, step=1)

# Using a select box to choose the first character for the script
characters = ['SpongeBob', 'Patrick', 'Squidward', 'Gary']
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