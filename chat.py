import json
import random

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from spell_check import correct_typos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "B-BOT"
# print(
#     "Hello, I am B-BOT, personal ChatBOT of Mr. Bibek. Let's chat! (type 'quit' or 'q' to exit)"  # NoQA
# )


def generate_response(sentence):
    # sentence = input("You: ")
    sentence = correct_typos(sentence)
    # print(sentence)
    if sentence.lower() == "quit" or sentence.lower() == "q":
        # Needs to quit
        pass

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.8:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        return (
            f"{bot_name}: Sorry, I do not understand... Can you be more "
            "specific on your question? You can ask about Bibek's skillset, "
            "experiences, portfolio, education, achievements "
            "and KAIST activities."
            "These are some sample questions: "
            "(I) Tell me about Bibek,\n"
            "(II) What skills does he have?,\n"
            "(III) What work experience does Bibek have?,\n"
            "(IV) What is Bibek's educational background?,\n"
            "(V) What awards has he won?,\n"
            "(VI) What projects has he completed? &\n"
            "(VII) How can I contact Bibek?"
        )


# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence.lower() == "quit" or sentence.lower() == "q":
#         break

#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.8:
#         for intent in intents["intents"]:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(
#             f"{bot_name}: Sorry, I do not understand... Can you be more "
#             "specific on your question? You can ask about Bibek's skillset, "
#             "experiences, portfolio, education, achievements "
#             "and KAIST activities."
#         )
