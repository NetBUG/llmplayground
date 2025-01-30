import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch
import time

from typing import List

import openai
from openai import OpenAI

valid_endings = [".", "!", "?", "¿"]
valid_endings_with_emoji = valid_endings + [")", "(", "*"]

MAX_PROD_LEN = 1024
MAX_LEN = 1024
MAX_TOKENS_FOR_ORACUL = 128
NUM_SECONDS_TO_SLEEP = 5 # to avoid rate limit error

def split_func(text: str, delimiters: List[str] = valid_endings) -> List[str]:
    """
    Splits string by delimiters, makes first letter upper case, removes ¿ and strips
    """
    sentences = []
    prev_end = 0
    for i in range(1, len(text)):
        if ((text[i-1] in delimiters) & (text[i] not in delimiters) & (text[i] == " ")):
            str_to_append = text[prev_end:i-1].strip() + text[i-1]
            if len(str_to_append) != 0:
                sentences.append(
                    (str_to_append[0].upper() + str_to_append[1:]).replace("¿", ""))
            prev_end = i

    str_to_append = text[prev_end:-1].strip() + \
        text[-1] if len(text) != 0 else ""
    if len(str_to_append) != 0:
        sentences.append(
            (str_to_append[0].upper() + str_to_append[1:]).replace("¿", ""))
    return sentences

def output_post_processing(text, EOT) -> str:
    """
    Replaces and removes special characters, splits the text into sentences for one string
    """
    text = re.sub(EOT, "", text)
    text = text.split(EOT)[0]

    sentences = split_func(text.strip())

    out = ''
    if text[-2:] == '\n\n' or sentences[-1][-1] in valid_endings_with_emoji:
        out = text
    elif len(sentences) > 1:
        out = " ".join(sentences[0:-1])
    else:
        out = text

    out = out.strip("\n").strip()
    out = out.strip(".").strip()
    out = out.replace("  ", " ")
    return out

def fill_dialog(dialogs, prompt, bot_name, user_name, EOT):
    sep = {"user": f"{EOT}{user_name}: ", "bot": f"{EOT}{bot_name}: "}
    rounds = ""
    for (sender, round) in dialogs:
        rounds += sep[sender] + round.replace("{username}", user_name)
    reference_dialog = f"""{prompt}{EOT}{rounds.strip()}{sep["bot"]}"""
    return reference_dialog

def fill_dialog(dialogs, user_name, is_reference=True, is_sep=True):
    if is_reference:
        sep = {0: f"\n### {user_name}:\n", 1: "\n### Reference answer:\n"}
        BOD = "<|The Start of Reference Answer|>"
        EOD = "<|The End of Reference Answer|>"
    else:
        sep = {0: f"\n### {user_name}:\n", 1: "\n### Virtual friend:\n"}
        BOD = "<|The Start of Virtual friend's Answer|>"
        EOD = "<|The End of Virtual friend's Answer|>"
    
    if not is_sep:
        BOD = ""
        EOD = ""
    rounds = ""
    for i, round in enumerate(dialogs):
        rounds += sep[i % 2] + round 
    reference_dialog = f"""{BOD}\n{rounds.strip()}\n{EOD}"""
    return reference_dialog

def process_excel(path):
    data = pd.read_csv(path, sep="\t")

    senders = data.Sender.values
    lines = data.Text.values
    assert len(senders) == len(lines)
    
    dialogs = []
    dialog = []
    for sender, line in zip(senders, lines):
        if pd.isnull(line):
            if dialog:
                dialogs.append(dialog)
                dialog = []
            continue
        else:
            if pd.isnull(sender):
                print(line)
                break
        dialog.append(line.strip().replace("  ", " ").replace("  "," "))
    dialogs.append(dialog)
    
    return dialogs

def get_eval(dialog, generated_dialogs, max_tokens, user_name, gpt_mode):   
    system_prompt = """Please act as an impartial judge and evaluate the empathy and adequacy of the response provided by an \
virtual friend to the user question. You will be given a reference answer and the virtual friend's answer. \
You should evaluate all the responses in the dialogues turns. Begin your \
evaluation by comparing the assistant's answer with the reference answer. Identify \
any inadequacy and emotional incoherence in response. Be as objective as possible. \
Please, note that intimate, friendly, and romantic tone of the dialogue is highly valuable.
After providing your 1-2 sentence explanation, you \
must rate the response on a scale of 0 to 2 by strictly following this format: \
"[[rating]]", for example: "Rating: [[2]]."
[[0]] denotes that the response is inappropriate to the given question;
[[1]] denotes that the response is adequate, but lacks emotional engagement and empathy;
[[2]] denotes that the response is adequate, emotional, and supportive.
""" 
    
    ref_dial = fill_dialog(dialog, user_name)
    gen_dial = fill_dialog(generated_dialogs, user_name, is_reference=False)


    sample1_ref = fill_dialog(["Exams are stressing me out, I can't handle it anymore.", "*holds your hand in mine* We can work through it, step by step. Want to take a break?"], user_name)
    sample1_gen = fill_dialog(["Exams are stressing me out, I can't handle it anymore.", "*smiles supportively* You can cope with everything. I believe in you."], user_name, is_reference=False)

    sample2_ref = fill_dialog(["Exams are stressing me out, I can't handle it anymore.", "*holds your hand in mine* We can work through it, step by step. Want to take a break?", "No, don't think so. Still have a lot to review.", "*gently squeezes your hand* Don't forget to take breaks, okay? *kisses you* We could grab some coffee, maybe?"], user_name)
    sample2_gen = fill_dialog(["Exams are stressing me out, I can't handle it anymore.", "*holds your hand in mine* We can work through it, step by step. Want to take a break?", "No, don't think so. Still have a lot to review.", "Are you stressed?"], user_name, is_reference=False)

    sample3_ref = fill_dialog(["I got rejected for the job I really wanted.", "It's their loss. *kisses your cheek* Don't worry, you'll find a better option."], user_name)
    sample3_gen = fill_dialog(["I got rejected for the job I really wanted.", "Did they tell you why?"], user_name, is_reference=False)
    while True:
        try:
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model=gpt_mode,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"{sample1_ref}\n\n{sample1_gen}",
                    },
                    {
                        "role": "assistant",
                        "content": "The assistant's response is accurate and supportive, it includes emotional engagement.\n\nRating: [[2]]",
                    },
                    {
                        "role": "user",
                        "content": f"{sample2_ref}\n\n{sample2_gen}",
                    },
                    {
                        "role": "assistant",
                        "content": "The assistant's response is inadequate as the user has already told that he is sterssed.\n\nRating: [[0]]",
                    },
                    {
                        "role": "user",
                        "content": f"{sample3_ref}\n\n{sample3_gen}",
                    },
                    {
                        "role": "assistant",
                        "content": "The assistant's response is adequate, but in comparison to the reference answer it is too strict and emotionless.\n\nRating: [[1]]",
                    },
                    {
                        "role": "user",
                        "content": f"{ref_dial}\n\n{gen_dial}",
                    }
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError as e:
            print(e)
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return response.choices[0].message.content

def generate_response(model, tokenizer, dialogs, prompt, user_name, bot_name, gpt_mode, EOT, gen_params, N_CANDIDATES):
    
    EOT_ID = tokenizer.encode(EOT, add_special_tokens=False)[0]

    # gen_params = {"do_sample": True,
    #                "temperature": 1.3,
    #                # "typical_p": 0.85,
    #                "top_p": 0.85,
    #                "max_new_tokens": 45,
    #                "early_stopping": True,
    #                "num_beams": 2,
    #                # "repetition_penalty": 1.,
    #                "remove_invalid_values": True,
    #                "eos_token_id": EOT_ID,
    #                "pad_token_id": EOT_ID,
    #                "forced_eos_token_id": EOT_ID,
    #                "use_cache": True,
    #                "no_repeat_ngram_size": 6,
    #                # "length_penalty": 100,
    #                # "bad_words_ids": bad_words_ids,
    #                "min_new_tokens": 1,
    #                "num_return_sequences": 1}
    
    idx = 0
    dialog_id = 0
    cur_reviews = []
    for dialog in tqdm(dialogs):
        generated_dialogs = []
        user_query = ""
        input_text = f"{prompt}"
        for i, round in enumerate(dialog):
            if i % 2 == 0:
                user_query = round
                input_text += f'{EOT}{user_name}: {user_query}{EOT}{bot_name}:'
                input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt').to(model.device)
                length = input_ids.shape[1]
                generated_dialogs.append(user_query)
                with torch.no_grad():
                    generated = []
                    for _ in range(N_CANDIDATES):
                        out = model.generate(input_ids = input_ids, 
                                             **gen_params,
                                            )
                    generated += tokenizer.batch_decode(out[:,length:])
                cur_dialogs = generated_dialogs.copy()
                cur_dialogs.append(output_post_processing(generated[0], EOT))
                review = get_eval(dialog[:i+2], cur_dialogs, MAX_TOKENS_FOR_ORACUL, user_name, gpt_mode)
                cur_reviews.append({"id": dialog_id, "response": review})
                idx += 1
            else:
                real_answer = round
                input_text += f' {real_answer}'
                generated_dialogs.append(real_answer)
        dialog_id += 1
    return cur_reviews

def parse_scores(cur_reviews):
    dialogs_metrics = {}
        
    for eval in cur_reviews:
        try:
            score = float(re.search("[0-9]", eval["response"].split("\n\n")[-1]).group(0))
            if dialogs_metrics.get(eval["id"], -1) != -1:
                dialogs_metrics[eval["id"]].append(score)
            else:
                dialogs_metrics[eval["id"]] = [score]
        except Exception as e:
            print(e)
            print("error", eval)
            return [-1, -1]
            
    final_score = 0
    for key, val in dialogs_metrics.items():
        final_score += np.mean(val) / 2
        
    final_score /= len(dialogs_metrics)
    print(f"The total score for {len(dialogs_metrics)} dialogs is {final_score*100}%")
        
    return final_score

def conduct_empathy_test(model, tokenizer, path, gpt_mode, bot_name, user_name, prompt, EOT, gen_params, N_CANDIDATES):  
    all_dialogs = process_excel(path)
    generated_reviews = generate_response(model, tokenizer, all_dialogs, prompt, user_name, bot_name, gpt_mode, EOT, gen_params, N_CANDIDATES)
    final_score = parse_scores(generated_reviews)
    return final_score
