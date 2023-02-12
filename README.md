# Hosted Link:
https://bibekbot.streamlit.app/

## Setup:
Clone the repo
```console
git clone git@github.com:bibekyess/Personal-Chatbot.git
cd Personal-Chatbot
```

### Create an environment using pipenv
Install pipenv if not installed. [Pipenv is required to install streamlit in macOS]
```console
pip3 install pipenv
```

Creates a new Pipenv environment using python-3.9 and activates it
```console
pipenv --python 3.9
pipenv shell
```

Installs streamlit in the recently created environment
```console
 pipenv install streamlit==1.11.1
```

Installs the dependencies mentioned in requirements.txt file
```console
pip install -r requirements.txt
```

Runs the b_bot app
```console
streamlit run b_bot.py
```

# For training:
```console
python train.py
```
This saves the checkpoint in 'data.pth' file. Then run the b_bot app
```console
streamlit run b_bot.py
```

Reference:
I referred to https://github.com/patrickloeber/pytorch-chatbot for simple implementation of contextual chatbot with interactions from terminals.
I referred to https://github.com/AI-Yash/st-chat for the beautiful chatbot interface.
