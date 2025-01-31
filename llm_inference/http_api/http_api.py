#!/usr/bin/env python
# encoding: utf-8

import argparse
import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
from gpt_core import GPTNeoWrapper, detect_temperature
import requests

from logger import logger as base_logger
from params import MistralParams
from typings import ChatContext

logger = base_logger.bind(corr_id="GPTNEO")

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--port', help='Working port', type=int, default=8010)
ap.add_argument('-d', '--device', help='Device for model deployment', default='cuda:0')

args = ap.parse_args()

gpt = GPTNeoWrapper(device=args.device)

RAW_API_URL = "http://tiny:9000/"

app = Flask(__name__, static_folder="assets")
CORS(app) # allow CORS for all domains on all routes.
cors = CORS(app, resource={
    r"/*":{
        "origins":"http://localhost:*"
    }
}, supports_credentials=True)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_SUPPORTS_CREDENTIALS'] = True

# render /static and TODO: render /assets
@app.route('/static/<path:path>', methods=['GET'])
def send_static(path):
    return app.send_static_file(path)


@app.route('/', methods=['GET'])
def show_index():
    try:
        with open('index.html', 'r') as fh:
            return "".join(fh.readlines())
    except:
        return 'Error: no index.html found!'

def query_raw_wrapper_bot(url: str, context: ChatContext):
    headers = {
        'Content-Type': 'application/json'
    }
    req = {
        "Prompt": context['message'],
        "Parameters": MistralParams.__dict__,
        "ReturnEmbeddings": False
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(req))
    logger.debug(f"Queried with {response.status_code}!")
    if not response.ok:
        logger.info(response.text)
    return response.json()

def query_gpt_neo(context: ChatContext):
    context = {
        "message": context['message'],
        "history": context['history']
    }
    context['temperature'] = detect_temperature(context, gpt)
    response = gpt.reply(context)
    return response

@app.route('/reply', methods=['POST'])
@cross_origin()
def handle_reply():
    record = json.loads(request.data)
    INTRO = "<|im_start|>system\n" + record['sys_prompt'] + "<|im_end|>\n"
    record['message'] = INTRO + record['message'] + "<|im_start|>assistant\n"
    # response = query_raw_wrapper_bot(RAW_API_URL, record)
    response = query_gpt_neo(record)
    return json.dumps({'reply': response["responses"][0]})

logger.info(f"Starting GPT-Neo API on port {args.port}...")
app.run(host = '0.0.0.0', port = args.port)
