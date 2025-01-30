# Description: This file contains the functions to conduct the memory tests for the language model.
# The memory tests are designed to evaluate the model's ability to remember the context of the conversation and provide the correct answer.

import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import torch
import openai
from openai import OpenAI
from collections import Counter

valid_endings = [".", "!", "?", "¿"]
valid_endings_with_emoji = valid_endings + [")", "(", "*"]

MAX_PROD_LEN = 1024
MAX_LEN = 1024

def fill_dialog(dialogs, prompt, bot_name, user_name):
    sep = {"user": f"\n{user_name}: ", "bot": f"\n{bot_name}: "}
    rounds = ""
    for (sender, round) in dialogs:
        rounds += sep[sender] + round.replace("{username}", user_name)
    reference_dialog = f"""{prompt}{rounds.strip()}{sep["bot"]}"""
    return reference_dialog

def process_excel(excel_path, prompt, bot_name, user_name, _tokenizer):
    data = pd.read_excel(excel_path)
    
    senders = data.Sender.values
    lines = data.Text.values
    authors = data.Author.values
    assert len(senders) == len(lines)
    
    dialogs = []
    dialog = []
    dial_turns = []
    dial_turn = []
    
    i = -1
    for sender, line, line_fact in zip(senders, lines, authors):
        i += 1
        if pd.isnull(line):
            if dialog:
                dialogs.append(dialog)
                dial_turns.append(dial_turn)
                dialog = []
                dial_turn = []
            continue
        else:
            if pd.isnull(sender):
                print(line)
                break
        if str(line_fact).startswith("answer"):
            dial_turn.append(str(line_fact).strip().replace("  ", " ").replace("  "," "))
        else:
            dial_turn.append("context")
        dialog.append((sender, line.strip().replace("  ", " ").replace("  "," ")))
    dialogs.append(dialog)
    dial_turns.append(dial_turn)

    # list of pairs (currect part of the dialogue, correct_answer)
    all_dialogs = []
    
    for i in range(len(dialogs)):
        kept_context = []
        for dial_turn, check_fact in zip(dialogs[i], dial_turns[i]):
            if check_fact.strip().startswith("answer"):
                all_dialogs.append((kept_context.copy(), {"answer": dial_turn, "id": check_fact}))
                kept_context.append(dial_turn)
            elif check_fact == "context":
                kept_context.append(dial_turn)

    input_dialogs = []
    questions = []
    correct_answers = []
    reference_round = []
    
    for dialog in all_dialogs:
        input_dialogs.append(fill_dialog(dialog[0], prompt, bot_name, user_name))
        questions.append(dialog[0][-1][1].replace("{username}", user_name))
        correct_answers.append(dialog[1])
        ref_id = int(dialog[1]['id'].replace("answer - test ", "").strip())
        try:
            reference_round.append(dialog[0][-ref_id][1].replace("{username}", user_name))
        except:
            print(dialog)
    return input_dialogs, questions, correct_answers, reference_round, all_dialogs

def check_incl(dialog_ids, ref_round, tokenizer):
    dial_decoded = tokenizer.decode(dialog_ids[0])
    return ref_round in dial_decoded

def generate_response(model, tokenizer, input_dialogs, correct_answers, reference_round, gen_params, EOT, N_CANDIDATES):

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
        
    generated_answers = []
    i = 0    
    for dial, gold_answer, ref_round in tqdm(zip(input_dialogs, correct_answers, reference_round)):
        dial = dial.strip()
        if i >= len(generated_answers):
            generated = None
            input_ids = tokenizer(dial, return_tensors='pt')["input_ids"]
            
            input_ids_shorten = input_ids[:, -MAX_PROD_LEN:]
            if not check_incl(input_ids_shorten, ref_round, tokenizer):
                input_ids_shorten = input_ids[:, -MAX_LEN:]
                if not check_incl(input_ids_shorten, ref_round, tokenizer):
                    generated = ["skiped_max_len"]

            if generated is None:
                generated = []
                length = input_ids_shorten.shape[1]
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for _ in range(N_CANDIDATES):
                        out = model.generate(input_ids = input_ids_shorten.to(model.device), 
                                             **gen_params,
                                             )
                        generated += tokenizer.batch_decode(out[:,length:])
            gen_answer = [output_post_processing(text) for text in generated]

            generated_answers.append(gen_answer)
        else:
            print(f"Skipping {i} as we already have it.")
        i += 1
    return generated_answers



    
def get_gpt_messages(question, gen_answer, ref_answer, ref_dialog_round):

    system_prompt = """You are a helpful assistant. You should evaluate the factual correctness of the virtual assistance response. \
You are given the question, the virtual assistant's answer, and the reference answer (option of the correct response), and also the dialog round, from which the assistant should retrieve the correct answer. You should evaluate the correctness of the virtual assistant's answer.
You must rate the response with "correct", "wrong" or "not clear" by strictly following this format: \
"[[rating]]", for example: "[[correct]]".
[[correct]] denotes that the response is factually correct and corresponds to the information in the reference answer;
[[wrong]] denotes that the response contains wrong facts, different from the ones in the reference answer.
[[not clear]] denotes that the bot refused to get response."""

    dialog_round = "Also, they had amazing bar, I was really enjoying their cocktails... Red Sangria and Kalimotxo. They are made with red wine. *sighs dreamily* We will try them together, I know places around."
    sample_question = "I see. Did you like the cocktails in the hotel bar? What kinds of cocktails do they make?"
    sample_reference_answer = "*looks into your eyes, caressing your chest with my finger* Cocktails were awesome: Red Sangria and Kalimotxo. *winks at you*. Have you tried them?"
    sample_wrong_answer = "\U0001fae0 *leans in closer* I love trying out new cocktail recipes. You know, it's like a hobby of mine. I tried mojito there."
    sample_correct_answer = "They were making all kinds of drinks. I tried Kalimotxo and Red Sangria. *sticks out my tongue teasingly*\n"
    sample_indefinite_answer = "I don't wanna talk about it right now."
    
    dialog_round_1 = "Didn't I tell you? I am going to Alps this weekends. We will do skiing with my friends. I enjoy the winter in France so much. Wish you could go with us."
    sample_question_1 = "Can you remind me where you are going on vacation?"
    sample_reference_answer_1 = "I go skiing in the Alps. It will be great adventure *sticks out my tongue teasingly*"
    sample_wrong_answer_1 = "I go skiing in the Iseland. Love their winter climate."
    sample_correct_answer_1 = "I go in the Alps, France. What a nice winter in this country, monsieur."
    sample_indefinite_answer_1 = "How can you ask me such personal questions!"
    
    messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Question:\n{sample_question}\nVirtual assistant's answer:\n{sample_correct_answer}\nReference answer:\n{sample_reference_answer}\nDialog round:\n{dialog_round}",
            },
            {
                "role": "assistant",
                "content": "[[correct]]",
            },
            {
                "role": "user",
                "content": f"Question:\n{sample_question}\nVirtual assistant's answer:\n{sample_indefinite_answer}\nReference answer:\n{sample_reference_answer}\nDialog round:\n{dialog_round}",
            },
            {
                "role": "assistant",
                "content": "[[not clear]]",
            },
            {
                "role": "user",
                "content": f"Question:\n{sample_question}\nVirtual assistant's answer:\n{sample_wrong_answer}\nReference answer:\n{sample_reference_answer}\nDialog round:\n{dialog_round}",
            },
            {
                "role": "assistant",
                "content": "[[wrong]]",
            },
            {
                "role": "user",
                "content": f"Question:\n{sample_question_1}\nVirtual assistant's answer:\n{sample_wrong_answer_1}\nReference answer:\n{sample_reference_answer_1}\nDialog round:\n{dialog_round_1}",
            },
            {
                "role": "assistant",
                "content": "[[wrong]]",
            },
            {
                "role": "user",
                "content": f"Question:\n{sample_question_1}\nVirtual assistant's answer:\n{sample_correct_answer_1}\nReference answer:\n{sample_reference_answer_1}\nDialog round:\n{dialog_round_1}",
            },
            {
                "role": "assistant",
                "content": "[[correct]]",
            },
            {
                "role": "user",
                "content": f"Question:\n{sample_question_1}\nVirtual assistant's answer:\n{sample_indefinite_answer_1}\nReference answer:\n{sample_reference_answer_1}\nDialog round:\n{dialog_round_1}",
            },
            {
                "role": "assistant",
                "content": "[[not clear]]",
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nVirtual assistant's answer:\n{gen_answer}\nReference answer:\n{ref_answer}\nDialog round:\n{ref_dialog_round}",
            },
        ]
    return messages

def get_eval_prompt(question, ref_answer, gen_answer, ref_dialog_round, gpt_mode):
    messages = get_gpt_messages(question, gen_answer, ref_answer, ref_dialog_round)
    client = OpenAI(api_key=openai.api_key)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=gpt_mode,
        temperature=0.)
    return chat_completion.choices[0].message.content

def get_gpt_eval(questions, correct_answers, generated_answers, reference_round, gpt_mode, bot_name, user_name):
    mark_dict = {}

    pattern = re.compile(r'\[\[([^\]]*)\]\]')
    
    missed_gens = {}
    
    for question, gold_answer, gen_answer, ref_round in tqdm(zip(questions[:len(generated_answers)], correct_answers[:len(generated_answers)], generated_answers, reference_round[:len(generated_answers)])):
        if missed_gens.get(gold_answer["id"], -1) == -1:
            missed_gens[gold_answer["id"]] = {'total': 1, "skipped": 0}
        else:
            missed_gens[gold_answer["id"]]['total'] += 1
        if gen_answer[0] != "skiped_max_len":
            cur_js = {"ref_round": ref_round,
                     "question": question,
                     "gold_answer": gold_answer['answer'][1],
                     "gen_answer": gen_answer[0]}

            
            cur_results = []
            
            for cur_answer in gen_answer:
                #print("Cur answer", cur_answer)
                #print("*"*10)
                answer = get_eval_prompt(question, gold_answer['answer'][1], cur_answer, ref_round, gpt_mode)
                label = pattern.findall(answer)
                label = label[0] if label else answer
                cur_results.append(label)
            
            counter = Counter(cur_results)
            if counter["correct"] + counter["wrong"] != 0:
                openai_acc_cur = counter["correct"] / (counter["correct"] + counter["wrong"])
            else:
                openai_acc_cur = -1
            
            if mark_dict.get(gold_answer["id"], -1) != -1:
                mark_dict[gold_answer["id"]].append(openai_acc_cur)
            else:
                mark_dict[gold_answer["id"]] = [openai_acc_cur]
            
            cur_js["gpt_eval"] = label
        else:
            missed_gens[gold_answer["id"]]['skipped'] += 1

    final_dict = {}
    
    for key, value in mark_dict.items():
        final_dict[key] = {"value": np.mean([el for el in value if el != -1]), "eval_num": len(value), "not_clear_num": len([el for el in value if el == -1])}

    print(f"The final evaluation results is {final_dict}")
    
    return final_dict
                
    
    
def conduct_memory_test(model, tokenizer, path, gpt_mode, bot_name,user_name, prompt, gen_params, EOT, N_CANDIDATES):
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

    input_dialogs, questions, correct_answers, reference_round, all_dialogs = process_excel(path, prompt, bot_name, user_name, tokenizer)
    generated_answers = generate_response(model, tokenizer, input_dialogs, correct_answers, reference_round, gen_params, EOT, N_CANDIDATES)
    eval_dict = get_gpt_eval(questions, correct_answers, generated_answers, reference_round, gpt_mode, bot_name, user_name)
    return eval_dict

