# SpongeBob-Text-Generation

![SpongeBob GIF](https://media2.giphy.com/media/nDSlfqf0gn5g4/giphy.gif)

This app uses [HuggingFace](https://huggingface.co/), an open repository and library to pretrained transformer models. A [smaller version of GPT-2](https://huggingface.co/distilgpt2) that was pretrained was finetuend on scripts of the show [SpongeBob SquarePants](https://www.kaggle.com/mikhailgaerlan/spongebob-squarepants-completed-transcripts), then an app was implemented and hosted with [Streamlit](https://streamlit.io/).

# App Details
[`launch_app.py`](https://github.com/Coldestadam/SpongeBob-Text-Generation/blob/main/launch_app.py) is the script that launches the app.

The link to the app is here: https://share.streamlit.io/coldestadam/spongebob-text-generation/main/launch_app.py

## Model and Data
1. Original Pretrained Model: https://huggingface.co/distilgpt2
2. Finetuned Model Repo: https://huggingface.co/Coldestadam/Breakout_Mentors_SpongeBob_Model
3. Kaggle Dataset: https://www.kaggle.com/mikhailgaerlan/spongebob-squarepants-completed-transcripts