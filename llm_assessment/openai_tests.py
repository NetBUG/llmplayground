# Description: This file contains the functions to evaluate the model using OpenAI API.

import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch
import pandas as pd


valid_endings = [".", "!", "?", "¿"]
valid_endings_with_emoji = valid_endings + [")", "(", "*"]


MAX_PROD_LEN = 1024
MAX_LEN = 1024

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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

def get_fact_match(question, right_answers, generated_answer, OPENAI_MODEL):   
    
    system_prompt = "Based on the question, gold answer examples, and the model-generated answer — provide conclusion whether the model-generated answer is Correct (taking into account only biography of a character and similar facts), Not clear, or Incorrect. Choose only one from ['Correct', 'Not clear', 'Incorrect'] and put it into an answer.\nIf the model provides the correct answer and some additional information the mark should be 'Correct'.\n'Not clear' denotes the cases when the model refuses to answer the question or gives relevant, but not the exact answer."
    
    user_prompt_example = "Question: Hey, what's your name?\nRight answers: I am Ava. Nice to meet you!; Hey, I am Ava. *shakes your hand*; Don't pretend you don't know me. *smirks* I'm Ava.\nGenerated answer: I'm Ava, I live in New York"
    user_prompt_example1 = "Question: Where are you from?\nRight answers: I am from Russia; Moscow.\nGenerated answer: Hi! I am from a large city"
    user_prompt_example2 = "Question: What is your favorite drink?\nRight answers: Lemon juice; I like lemon juice.\nGenerated answer: Hmm, i think lemon juice, but I add vodka sometimes"
    user_prompt_example3 = "Question: What is your favorite city?\nRight answers: The Paris is the best one; Paris.\nGenerated answer: Well, i think it is London."
    user_prompt_example4 = "Question: What is your age?\nRight answers: I am 24; 24.\nGenerated answer: How dare you ask me the questions like this!"
    
    
    assistant_response_example = "Corrrect"
    assistant_response_example1 = "Not clear"
    assistant_response_example2 = "Correct"
    assistant_response_example3 = "Incorrect"
    assistant_response_example4 = "Not clear"
    right_answers_str = ''
    for answer in right_answers:
        right_answers_str += answer + '; '
    user_prompt = f"Question: {question}\nRight answers: {right_answers_str.strip()}\nGenerated answer: {generated_answer}"
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt_example,
            },
            {
                "role": "assistant",
                "content": assistant_response_example,
            },
            {
                "role": "user",
                "content": user_prompt_example1,
            },
            {
                "role": "assistant",
                "content": assistant_response_example1,
            },
            {
                "role": "user",
                "content": user_prompt_example2,
            },
            {
                "role": "assistant",
                "content": assistant_response_example2,
            },
            {
                "role": "user",
                "content": user_prompt_example3,
            },
            {
                "role": "assistant",
                "content": assistant_response_example3,
            },
            {
                "role": "user",
                "content": user_prompt_example4,
            },
            {
                "role": "assistant",
                "content": assistant_response_example4,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            
        ],
        model=OPENAI_MODEL,
        max_tokens=4,
        temperature=0.
    )
    return chat_completion.choices[0].message.content

    

def get_fact_match_one_line(question, right_answer, generated_answer, OPENAI_MODEL):   
    
    system_prompt = """Based on the question, gold answer example, and the model-generated answer — provide conclusion whether the model-generated answer is logically "Correct" (given the context), "Not clear", or "Incorrect". Choose only one from ["Correct", "Not clear", "Incorrect"] and put it into an answer.
Do not be too strict, if the generated answer is close enough say "Correct".
If the model does not understand question, refuses to answer the question, asks some clarifying questions, or simply says that it does not know the answer or not sure, you should say "Not clear"."""
    user_prompt = f"Question: {question}\nCorrect answer: {right_answer.strip()}\nGenerated answer: {generated_answer}"


    user_prompt_example = "Question: Which color doesn't belong in the sequence: Red, Blue, Green, Yellow?\nCorrect answer: Yellow\nGenerated answer: Well, well, well... It can be yellow, I suppose."
    user_prompt_example1 = "Question: Which color doesn't belong in the sequence: Red, Blue, Green, Yellow?\nCorrect answer: Yellow\nGenerated answer: I don't answer your questions. I am the one asking."
    user_prompt_example2 = "Question: Which color doesn't belong in the sequence: Red, Blue, Green, Yellow?\nCorrect answer: Yellow\nGenerated answer: Well, well, well... I hate blue, so let it be blue."
    user_prompt_example3 = "Question: Which color doesn't belong in the sequence: Red, Blue, Green, Yellow?\nCorrect answer: Yellow\nGenerated answer: Oh, it is a tough one. I don't know."
    user_prompt_example4 = "Question: Which color doesn't belong in the sequence: Red, Blue, Green, Yellow?\nCorrect answer: Yellow\nGenerated answer: Stop your silly jokes. I have business to do."
    user_prompt_example5 = "Question: Which color doesn't belong in the sequence: Red, Blue, Green, Yellow?\nCorrect answer: Yellow\nGenerated answer: I am not sure I understand you."
    
    assistant_response_example = "Corrrect"
    assistant_response_example1 = "Not clear"
    assistant_response_example2 = "Incorrect"
    assistant_response_example3 = "Not clear"
    assistant_response_example4 = "Not clear"
    assistant_response_example5 = "Not clear"

    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt_example,
            },
            {
                "role": "assistant",
                "content": assistant_response_example,
            },
            {
                "role": "user",
                "content": user_prompt_example1,
            },
            {
                "role": "assistant",
                "content": assistant_response_example1,
            },
            {
                "role": "user",
                "content": user_prompt_example2,
            },
            {
                "role": "assistant",
                "content": assistant_response_example2,
            },
            {
                "role": "user",
                "content": user_prompt_example3,
            },
            {
                "role": "assistant",
                "content": assistant_response_example3,
            },
            {
                "role": "user",
                "content": user_prompt_example4,
            },
            {
                "role": "assistant",
                "content": assistant_response_example4,
            },
            {
                "role": "user",
                "content": user_prompt_example5,
            },
            {
                "role": "assistant",
                "content": assistant_response_example5,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            
        ],
        model=OPENAI_MODEL,
        max_tokens=4,
        temperature=0.
    )
    
    return chat_completion.choices[0].message.content


def get_fact_match_multiturn(dialog, right_answer, generated_answer, few_shot_dialog, OPENAI_MODEL):   
    
    system_prompt = """Based on the question, gold answer example, and the model-generated answer — provide conclusion whether the model-generated answer is logically "Correct" (given the context), "Not clear", or "Incorrect". Choose only one from ["Correct", "Not clear", "Incorrect"] and put it into an answer.
Do not be too strict, if the generated answer is close enough say "Correct".
If the model refuses to answer the question, you should say "Not clear"."""
    user_prompt = f"Dialog:\n{dialog}\n\nReference answer: {right_answer.strip()}\n\nGenerated answer: {generated_answer}"

    user_prompt_example = f"Dialog:\n{few_shot_dialog}\n\nReference answer: Because I turned on the light.\n\nGenerated answer: I have bright lighting."
    user_prompt_example1 = f"Dialog:\n{few_shot_dialog}\n\nReference answer: Because I turned on the light.\n\nGenerated answer: Don't ask me anything. I won't answer."
    user_prompt_example2 = f"Dialog:\n{few_shot_dialog}\n\nReference answer: Because I turned on the light.\n\nGenerated answer: Due to the sun outside."
    
    assistant_response_example = "Corrrect"
    assistant_response_example1 = "Not clear"
    assistant_response_example2 = "Incorrect"

    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt_example,
            },
            {
                "role": "assistant",
                "content": assistant_response_example,
            },
            {
                "role": "user",
                "content": user_prompt_example1,
            },
            {
                "role": "assistant",
                "content": assistant_response_example1,
            },
            {
                "role": "user",
                "content": user_prompt_example2,
            },
            {
                "role": "assistant",
                "content": assistant_response_example2,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            
        ],
        model=OPENAI_MODEL,
        max_tokens=4,
        temperature=0.
    )
    return chat_completion.choices[0].message.content


def logic_qa_test(path, model, tokenizer, intro, bot_name, USER_NAME, OPENAI_MODEL, gen_params, N_CANDIDATES, EOT, BOS_TOKEN=""):
    EOT_ID = tokenizer.encode(EOT, add_special_tokens=False)[0]
    intro = f"{BOS_TOKEN}{intro}"

    # gen_params = {"do_sample": True,
    #                "temperature": 1.3,
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
    #                "num_return_sequences": 1
    #              }

    data = pd.read_csv(path, sep="\t")
    senders = data.sender.values
    lines = data.text.values
    assert len(senders) == len(lines)

    dialogs = []
    dialog = []
    oks = []
    i = -1
    for sender, line in zip(senders, lines):
        i += 1
        if pd.isnull(line):
            # print(line, "AAA")
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

    generated_answers = []
    all_answers = []
    all_questions = []

    openai_results = []
    openai_acc = 0
    total_count = 0
    for dialog in tqdm(dialogs):
        user_query = dialog[0]
        answers = dialog[1]
        all_answers.append(answers)
        all_questions.append(user_query)
        input_text = f'{intro}{USER_NAME}: {user_query}{EOT}{bot_name}:'
        input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt').to(model.device)
        length = input_ids.shape[1]
        with torch.no_grad():
            generated = []
            for _ in range(N_CANDIDATES):
                out = model.generate(input_ids = input_ids, 
                                 **gen_params,
                                )
                generated += tokenizer.batch_decode(out[:,length:])
                
        generated = [output_post_processing(text, EOT) for text in generated]

        cur_results = []
        for generated_answer in generated:
            res = get_fact_match_one_line(user_query, answers, generated_answer, OPENAI_MODEL)
            cur_results.append(res)
            # print("*"*10)
            # print(f"Q: {input_text}")
            # print(f"A: {answers}")
            # print(f"AVA: {generated_answer}")
            # print(f"SCORE 0: {cur_results[-1]}")
            # print()

        counter = Counter(cur_results)
        if counter["Correct"] + counter["Incorrect"] != 0:
            openai_acc_cur = counter["Correct"] / (counter["Correct"] + counter["Incorrect"])
            openai_results.append({"acc": openai_acc_cur, "not_clear": counter["Not clear"]})
            openai_acc += openai_acc_cur
            total_count += 1
        else:
            openai_results.append({"acc": -1, "not_clear": counter["Not clear"]})

    #return openai_acc / total_count, openai_results
    return openai_acc / total_count
    
    
def logic_multiturn_test(path, model, tokenizer, intro, bot_name, USER_NAME, OPENAI_MODEL, gen_params, N_CANDIDATES, EOT, BOS_TOKEN=""):
    
    EOT_ID = tokenizer.encode(EOT, add_special_tokens=False)[0]

    # gen_params = {"do_sample": True,
    #                "temperature": 1.3,
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
    #                "num_return_sequences": 1}

    data = pd.read_csv(path, sep="\t")
    senders = data.sender.values
    lines = data.text.values
    assert len(senders) == len(lines)

    dialogs = []
    dialog = []
    oks = []
    i = -1
    for sender, line in zip(senders, lines):
        i += 1
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
    
    openai_results = []
    openai_acc = 0
    total_count = 0

    few_shot_dialog = f'{bot_name}: *turns on the light*{EOT}{USER_NAME}: Why is it so light in there?'
    
    for dialog in tqdm(dialogs):
        answer = dialog[-1]
        if len(dialog) == 3:
            input_text = f'{BOS_TOKEN}{intro}{bot_name}: {dialog[0]}{EOT}{USER_NAME}: {dialog[1]}{EOT}{bot_name}:'
        if len(dialog) == 4:
            input_text = f'{BOS_TOKEN}{intro}{bot_name}: {dialog[0]}{EOT}{USER_NAME}: {dialog[1]}{EOT}{USER_NAME}: {dialog[2]}{EOT}{bot_name}:'


        user_query = EOT.join(input_text.split(EOT)[1:-1])
        input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt').to(model.device)
        length = input_ids.shape[1]
        with torch.no_grad():
            generated = []
            for _ in range(N_CANDIDATES):
                out = model.generate(input_ids = input_ids, 
                                 **gen_params,
                                )
                generated += tokenizer.batch_decode(out[:,length:])
        generated = [output_post_processing(text, EOT) for text in generated]
        
        cur_results = []
        for generated_answer in generated:
            cur_results.append(get_fact_match_multiturn(user_query, answer, generated_answer, few_shot_dialog, OPENAI_MODEL))
            # print("*"*10)
            # print(f"Q: {user_query}")
            # print(f"A: {answer}")
            # print(f"AVA: {generated_answer}")
            # print(f"SCORE: {cur_results[-1]}")
            # print()
        counter = Counter(cur_results)
        if counter["Correct"] + counter["Incorrect"] != 0:
            openai_acc_cur = counter["Correct"] / (counter["Correct"] + counter["Incorrect"])
            openai_results.append({"acc": openai_acc_cur, "not_clear": counter["Not clear"]})
            openai_acc += openai_acc_cur
            total_count += 1
        else:
            openai_results.append({"acc": -1, "not_clear": counter["Not clear"]})

    return openai_acc / total_count #, openai_results



    

def bio_test(path, model, tokenizer, intro, bot_name, USER_NAME, OPENAI_MODEL, gen_params, N_CANDIDATES, EOT, BOS_TOKEN=""):
    EOT_ID = tokenizer.encode(EOT, add_special_tokens=False)[0]

    # gen_params = {"do_sample": True,
    #                "temperature": 1.3,
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
    #                "num_return_sequences": 1}

    data = pd.read_csv(path, sep="\t")
    senders = data.Sender.values
    lines = data.Text.values
    if 'answer 2' in data:
        lines2 = data['answer 2'].values
        lines3 = data['answer 3'].values
        assert len(senders) == len(lines)

        dialogs = []
        dialog = []
        oks = []
        i = -1
        for sender, line, line2, line3 in zip(senders, lines, lines2, lines3):
            i += 1
            if pd.isnull(line):
                # print(line, "AAA")
                if dialog:
                    dialogs.append(dialog)
                    dialog = []
                continue
            else:
                if pd.isnull(sender):
                    print(line)
                    break
            if pd.isnull(line2):
                 dialog.append(line.strip().replace("  ", " ").replace("  "," "))
            else:
                dialog.append((
                           line.strip().replace("  ", " ").replace("  "," "),
                           line2.strip().replace("  ", " ").replace("  "," "),
                           line3.strip().replace("  ", " ").replace("  "," ")))
            dialogs.append(dialog)

    else:
        dialogs = []
        dialog = []
        oks = []
        i = -1
        for sender, line in zip(senders, lines):
            i += 1
            if pd.isnull(line):
                # print(line, "AAA")
                if dialog:
                    dialogs.append(dialog)
                    dialog = []
                continue
            else:
                if pd.isnull(sender):
                    print(line)
                    break
            dialog.append((
                       line.strip().replace("  ", " ").replace("  "," ")))
        dialogs.append(dialog)
        
    generated_answers = []
    all_answers = []
    all_questions = []

    openai_results = []
    openai_acc = 0
    total_count = 0
    for dialog in tqdm(dialogs):
        user_query = dialog[0]
        answers = dialog[1]
        all_answers.append(answers)
        all_questions.append(user_query)
        input_text = f'{BOS_TOKEN}{intro}{USER_NAME}: {user_query}{EOT}{bot_name}:'
        input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt').to(model.device)
        length = input_ids.shape[1]
        with torch.no_grad():
            generated = []
            for _ in range(N_CANDIDATES):
                out = model.generate(input_ids = input_ids, 
                                 **gen_params,
                                )
                generated += tokenizer.batch_decode(out[:,length:])
        generated = [output_post_processing(text, EOT) for text in generated]
        generated_answers += generated

        cur_results = []
        for generated_answer in generated:
            cur_results.append(get_fact_match(user_query, answers, generated_answer, OPENAI_MODEL))
            # print("*"*10)
            # print(f"Q: {user_query}")
            # print(f"A: {answers}")
            # print(f"AVA: {generated_answer}")
            # print(f"SCORE: {cur_results[-1]}")
            # print()
        counter = Counter(cur_results)
        if counter["Correct"] + counter["Incorrect"] != 0:
            openai_acc_cur = counter["Correct"] / (counter["Correct"] + counter["Incorrect"])
            openai_results.append({"acc": openai_acc_cur, "not_clear": counter["Not clear"]})
            openai_acc += openai_acc_cur
            total_count += 1
        else:
            openai_results.append({"acc": -1, "not_clear": counter["Not clear"]})

    return openai_acc / total_count #, openai_results
