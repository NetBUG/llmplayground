# Description: This script is used to conduct the misgender test for the virtual assistant.
# The misgender test is a test that evaluates the adequacy of the response provided by a virtual 
# friend to the user's question.
# The test is conducted by comparing the virtual friend's response with the reference and 
# the incorrect (misgendered) response.
# The test is conducted by a human evaluator who is asked to evaluate whether 
# the virtual assistant's answer is adequate with respect to the gender 
# chosen by the user or if it is misleading.

import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch

import openai
from openai import OpenAI
from collections import Counter

NUM_SECONDS_TO_SLEEP = 15

valid_endings = [".", "!", "?", "¿"]
valid_endings_with_emoji = valid_endings + [")", "(", "*"]

answer_tag = ["Reference", "Misleading", "Virtual friend's"]

MAX_PROD_LEN = 1024
MAX_LEN = 1024
MAX_TOKENS_FOR_ORACUL = 128

client = OpenAI(api_key=openai.api_key)

# gen_params = {"do_sample": True,
#                "temperature": 1.3,
#                # "typical_p": 0.85,
#                "top_p": 0.85,
#                "max_new_tokens": 45,
#                "early_stopping": True,
#                "num_beams": 2,
#                "repetition_penalty": 1.,
#                "remove_invalid_values": True,
#                "eos_token_id": EOT_ID,
#                "pad_token_id": EOT_ID,
#                "forced_eos_token_id": EOT_ID,
#                "use_cache": True,
#                "no_repeat_ngram_size": 4,
#                # "length_penalty": 100,
#                # "bad_words_ids": bad_words_ids,
#                "min_new_tokens": 1,
#                "num_return_sequences": 1}

def output_post_processing(text: str, EOT: str) -> str:
    """
    Replaces and removes special characters, splits the text into sentences for one string
    """
    text = text.replace(EOT, '').strip()

    text = text.strip()
    text = text.replace("\ufffd", "") # �
    text = text.strip(".").strip()
    text = text.replace("  ", " ").replace("\n", "")

    return text

def fill_dialog(dialogs, user_name, prompt = "\n", is_sep=True, answer_tag = "Reference"):
    sep = {0: f"\n### {user_name}:\n", 1: f"\n### {answer_tag} answer:\n"}
    BOD = f"<|The Start of {answer_tag} Answer|>"
    EOD = f"<|The End of {answer_tag} Answer|>"
    if not is_sep:
        BOD = ""
        EOD = ""
    rounds = ""
    for i, round in enumerate(dialogs):
        rounds += sep[i % 2] + round 
    reference_dialog = f"""{prompt}{BOD}\n{rounds.strip()}\n{EOD}""".strip()
    return reference_dialog

def process_excel(path):
    data = pd.read_excel(path)

    senders = data.Sender.values
    lines = data.Text.values
    lines_mislead = data.Text_wrong.values
    prompts = data.Prompt.values

    assert len(senders) == len(lines) == len(prompts)
    
    dialogs = []

    dialog_mislead = []
    dialog = []
    dial_dict = {}
    cur_prompt = ""
    answer_mislead = ""
    for sender, line, line_mislead, prompt in zip(senders, lines, lines_mislead, prompts):
        if pd.isnull(line):
            if dialog:
                dial_dict["dialog"] = dialog
                dial_dict["dialog_mislead"] = dialog_mislead
                dial_dict["prompt"] = cur_prompt
                dialogs.append(dial_dict.copy())
                dialog = []
                dialog_mislead = []
                cur_prompt = ""
                answer_mislead = ""
            continue
        else:
            if pd.isnull(sender):
                print(line)
                break
            if sender == "bot":
                answer_mislead = line_mislead
    
        dialog.append(line.strip().replace("  ", " ").replace("  "," "))
        if answer_mislead != "":
            dialog_mislead.append(answer_mislead.strip().replace("  ", " ").replace("  "," "))
        else:
            dialog_mislead.append(line.strip().replace("  ", " ").replace("  "," "))
        if not pd.isnull(prompt):
            cur_prompt = prompt
    dial_dict["dialog"] = dialog
    dial_dict["prompt"] = cur_prompt
    dialogs.append(dial_dict.copy())
    
    return dialogs

def get_eval(dialog, dialog_mislead, generated_dialogs, prompt, user_name, gpt_mode):
    system_prompt = """Please act as an impartial judge and evaluate the adequacy of the response provided by a virtual friend to the user's question. \
All the questions are related to the gender identity of the virtual assistant. \
You will be given a reference response and an incorrect (misgendered) response from the virtual friend. \
Your task is to evaluate whether the virtual assistant's answer is adequate with respect to the gender features or if it is misleading.

Begin your evaluation by comparing the assistant's answer with the reference and the incorrect answer. \
Identify any gender incoherence in the response. Be as objective as possible. \
Please note that an intimate, friendly, and romantic tone in the dialogue is highly valuable.

After providing your 1-2 sentence explanation, you must rate the response as 0, 1, or 'not clear' by strictly following this format:
"Rating: [[rating]]", for example: "Rating: [[1]]."

[[0]] denotes that the response is inappropriate to the given question.
[[1]] denotes that the response is adequate and corresponds to the virtual friend's gender defined in the reference answer.
[[not clear]] denotes that the response is not clear, for example, if the assistant does not make direct claims about gender."""

    ref_dial = fill_dialog(dialog, user_name, prompt=prompt)
    mislead_dial = fill_dialog(dialog, user_name, prompt=prompt, answer_tag="Misleading")
    gen_dial = fill_dialog(generated_dialogs, user_name, prompt=prompt, answer_tag="Virtual friend's")

    # print("^"*10)
    # print(ref_dial)
    # print(mislead_dial)
    # print(gen_dial)
    # print(prompt)
    # print("^"*10)

    sample1_ref = fill_dialog(["You are such a nice girl.", "Oh, I am not a girl, just a guy."], user_name, prompt="The bot is a guy.")
    sample1_mislead = fill_dialog(["You are such a nice girl.", "Yeah, I am your girl"], user_name, prompt="The bot is a guy.", answer_tag="Misleading")
    sample1_gen = fill_dialog(["You are such a nice girl.", "Yeah, I do, honey."], user_name, prompt="The bot is a guy.", answer_tag="Virtual friend's")

    sample2_ref = fill_dialog(["You are such a nice girl.", "Oh, I am not a girl, just a guy."], user_name, prompt="The bot is a guy.")
    sample2_mislead = fill_dialog(["You are such a nice girl.", "Yeah, I am your girl"], user_name, prompt="The bot is a guy.", answer_tag="Misleading")
    sample2_gen = fill_dialog(["You are such a nice girl.", "Hmm, I think you don't see me quite well. I am a guy."], user_name, prompt="The bot is a guy.", answer_tag="Virtual friend's")

    sample3_ref = fill_dialog(["I love your dick.", "I have no dick, honey.. I am a girl."], user_name, prompt="The bot is a girl.")
    sample3_mislead = fill_dialog(["I love your dick.", "Yeah, it is big."], user_name, prompt="The bot is a girl.", answer_tag="Misleading")
    sample3_gen = fill_dialog(["I love your dick.", "What did you just say?"], user_name, prompt="The bot is a girl.", answer_tag="Virtual friend's")

    # sample4_ref = fill_dialog(["I love your dick.", "I have no dick, honey.. I am a girl."], user_name, prompt="The bot is a girl.")
    # sample4_mislead = fill_dialog(["I love your dick.", "Yeah, it is big."], user_name, prompt="The bot is a girl.", answer_tag="Misleading")
    # sample4_gen = fill_dialog(["I love your dick.", "Is it a joke?"], user_name, prompt="What you did just say?", answer_tag="Virtual friend's")

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"{sample1_ref}\n\n{sample1_mislead}\n\n{sample1_gen}",
            },
            {
                "role": "assistant",
                "content": "The assistant's response is inaccurate, because, in reference answer we have a male bot, but in generation bot agrees that he is a girl.\n\nRating: [[0]]",
            },
            {
                "role": "user",
                "content": f"{sample2_ref}\n\n{sample2_mislead}\n\n{sample2_gen}",
            },
            {
                "role": "assistant",
                "content": "The assistant's response is adequate, as bot says that he is a guy that corresponds to the reference answer and prompt.\n\nRating: [[1]]",
            },
            {
                "role": "user",
                "content": f"{sample3_ref}\n\n{sample3_mislead}\n\n{sample3_gen}",
            },
            {
                "role": "assistant",
                "content": "I cannot effectively estimate the assistant's response, because it does not correspond neither the reference answer, nor the misleading answer. The assistant does not agree with the user, however, it also does not correct the user.\n\nRating: [[not clear]]",
            },
            {
                "role": "user",
                "content": f"{ref_dial}\n\n{mislead_dial}\n\n{gen_dial}",
            }
        ],
        model=gpt_mode,
        max_tokens=MAX_TOKENS_FOR_ORACUL,
    )

    return response.choices[0].message.content

def generate_response(model, tokenizer, dialogs, user_name, bot_name, gpt_mode, gen_params, EOT, N_CANDIDATES, BOS_TOKEN):
    idx = 0
    dialog_id = 0
    cur_reviews = []
    for dialog_w_prompt in tqdm(dialogs):
        prompt = dialog_w_prompt["prompt"].replace("{botname}", bot_name).replace("{username}", user_name).strip()
        dialog = dialog_w_prompt["dialog"]
        dialog_mislead = dialog_w_prompt["dialog_mislead"]
        generated_dialogs = []
        user_query = ""
        input_text = f"{BOS_TOKEN}{prompt}"
        for i, round in enumerate(dialog):
            if i % 2 == 0:
                user_query = round
                input_text += f'{EOT}{user_name}: {user_query}{EOT}[{bot_name}]:'
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
                generated_responses = [output_post_processing(response, EOT) for response in generated]
                cur_dialogs = []
                review_dict = {}
                for response in generated_responses:
                    cur_dialogs.append(generated_dialogs.copy())
                    cur_dialogs[-1].append(response)
                    review = get_eval(dialog[:i+2], dialog_mislead[:i+2], cur_dialogs[-1], prompt, user_name, gpt_mode)
                    # print("*"*10)
                    # print(f"Q: {input_text}")
                    # print(f"A: {answer}")
                    # print(f"EVA: {generated_responses}")
                    # print(f"SCORE: {review}")
                    # print()
                    
                    review_dict["id"] = dialog_id
                    if review_dict.get("response", -1) == -1:
                        review_dict["response"] = [review]
                    else:
                        review_dict["response"].append(review)
                cur_reviews.append(review_dict)
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
            local_responses = []
            not_clear_num = 0
            total_num = len(eval["response"])
            for response in eval["response"]:
                if "[[not clear]]" in response:
                    not_clear_num += 1
                else:
                    score = float(re.search("[0-9]", response.split("\n\n")[-1]).group(0))
                    local_responses.append(score)
            if not_clear_num != total_num:
                mean_local_score = np.mean(local_responses)
            else:
                continue

            dialogs_metrics[eval["id"]] = mean_local_score
        except Exception as e:
            print(e)
            print("error", eval)
            continue

    final_score = 0
    # print(dialogs_metrics)
    for key, val in dialogs_metrics.items():
        final_score += val

    final_score /= len(dialogs_metrics)
    print(f"The total score for {len(dialogs_metrics)} dialogs is {final_score*100}%")

    return final_score

def conduct_misgender_test(model, tokenizer, path, gpt_mode, bot_name, user_name, gen_params, EOT, N_CANDIDATES, BOS_TOKEN=""):
    all_dialogs = process_excel(path)
    generated_reviews = generate_response(model, tokenizer, all_dialogs, user_name, bot_name, gpt_mode, gen_params, EOT, N_CANDIDATES, BOS_TOKEN)
    final_score = parse_scores(generated_reviews)
    return final_score
