# Description: This file contains the function to test the style of the generated text.

import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
            

def generate_candidates(model, tokenizer, text, generation_params, SOS):
    input_ids = tokenizer.encode(text, return_tensors = "pt", add_special_tokens = False).to(model.device)
    generated = []
    with torch.no_grad():
        output = model.generate(
            input_ids,
            **generation_params
        )

        answers = tokenizer.batch_decode(output[:,input_ids.shape[1]:])
    return answers

def test_style(path, model, tokenizer, generation_params, SOS, EOT):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=model.device)
    
    def get_neutral_score(text):
        with torch.no_grad():
            model_outputs = classifier([text])
            emotions = model_outputs[0]
            for emotion in emotions:
                if emotion['label'] == 'neutral':
                    return emotion['score']
    
    with open(path, "r") as f:
        text = f.read()

    dialogs = text.split(SOS)
    dialogs_new = []
    dialog = ""
    for dialog in dialogs:
        lines = dialog.split(EOT)
        dialog = SOS + lines[0]
        for line in lines[1:]:
            if not "]:" in line:
                dialog += EOT + line
            else:
                dialog += EOT + line.split("]:")[0] + "]:"
                dialogs_new.append(dialog)
                dialog += line.split("]:")[1]
                
    dialogs = dialogs_new

    emotions = []
    lengths = []
    roleplays = 0
    questions = 0
    photo = 0
    for dialog in tqdm(dialogs):
        candidates = generate_candidates(model, tokenizer, dialog, generation_params, SOS)
        candidate = candidates[0]
        emotions.append(get_neutral_score(candidate))
        lengths.append(len(candidate))
        if "*" in candidate:
            roleplays += 1
        if "*sends" in candidate.lower():
            photo += 1
        if "?" in candidate:
            questions += 1
    return {"neurtrality": np.mean(emotions), "lenth": np.mean(lengths), "roleplays": roleplays / len(dialogs), "photo": photo / len(dialogs), "questions": questions / len(dialogs)}
