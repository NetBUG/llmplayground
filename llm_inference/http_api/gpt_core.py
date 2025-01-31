from datetime import datetime
import math
from logger import debug, info
from params import DEFAULT_PROFILE, MODEL_NAME, MODEL_TIMEOUT, MAX_CONTEXT_TURNS, GPTNeoParams

from json import dumps, loads
import time
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import torch

from logger import logger as base_logger
from typings import ChatContext

logger = base_logger.bind(corr_id="GPT_CORE")

DEFAULT_DEVICE = 'cpu'

blacklist = ['profile', 'Person', 'bot', 'Bot', 'horny', '|pad|']

def strip_prefix(msg):
    if msg.lower().startswith('bot:'):
        return msg[4:]
    if msg.lower().startswith('user:'):
        return msg[5:]
    if msg.lower().startswith('person:'):
        return msg[7:]
    return msg


def estimate_candidates(i):
    return min(10 + i * 7, 10 + math.ceil(i + 3*math.log(i + 2)))

def detect_auto_message(msg: ChatMessage):
    if msg.source == 'user':
        return True
    try:
        label = loads(msg.label)
        if 'OperatorId' in label.keys():
            return False
    except:
        pass
    return False


def insert_profile(hist: list, profile, cnt=12):
    pos = 0
    for i in range(math.ceil(len(hist) / 12)):
        pos = i * 12
        if len(hist) - pos < 2:
            pos = len(hist) - 2
        hist.insert(pos, profile)
    return hist


def encode_context(context: ChatContext, profile_freq=12):
    hstr = context.message
    hist = [] # make_history(context)
    if context.history and len(context.history) > 0:
        hist += [('Person:' if detect_auto_message(m) else 'Bot:') + m.message for m in context.history]

    profile = '' # context.personality
    if not profile:
        gender = '21'
        name = 'Ava'
        age = ''
        profile = DEFAULT_PROFILE
        if context.context and context.context.bot and context.context.bot.name:
            name = context.context.bot.name
            if context.context.bot.gender == 'mal':
                gender = 'male'
            elif context.context.bot.gender == 'fem':
                gender = 'female'
            else:
                logger.info('Unknown gender: %s' % str(context.context.bot.gender.gender))
                gender = ''
            if context.context.bot.birthday:
                dto = datetime.strptime(context.context.bot.birthday, '%Y-%m-%dT%H:%M:%SZ')
                delta =  datetime.now() - dto
                age = ', %d y.o' % math.floor(delta.days / 365)
        profile = DEFAULT_PROFILE.format(name=name, gender=gender, age=age)
    if not profile.startswith('Bot profile:'):
        profile = 'Bot profile:' + profile
    if not profile.strip().endswith('.'):
        profile = profile.strip() + '.'
    history = [c[:c.find('<')] if '<' in c else c for c in hist] + ['Person:%s' % hstr]
    history = hist[-MAX_CONTEXT_TURNS:] + ['Person:%s' % hstr]

    # No bot profile for greetings
    if len(history) < 0:    # 4 
        return history

    # insert bot profile in each 12 sentence!
    inputs = insert_profile(history, profile, profile_freq)
    return inputs


class GPTNeoWrapper():
    model = None
    tokenizer = None
    acceleration = DEFAULT_DEVICE
    params = GPTNeoParams()

    '''
    Initialize the NLP pipeline for GPT (transformers package)
    '''
    def __init__(self, model_name=MODEL_NAME, device='cuda:0'):
        logger.debug('GPU available: %s' % torch.cuda.is_available())
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        try:
            if not torch.cuda.is_available():
                raise Exception('No GPU available, creating CPU-based pipeline')
            self.acceleration = device
            logger.info('Using %s...' % self.acceleration)
            self.model.half().to(self.acceleration)
            logger.debug('Moved to GPU!')
        except Exception as e:
            self.acceleration = DEFAULT_DEVICE
            logger.debug(e)

    def calculate_loss(self, input_ids):
        input_ids = input_ids

        with torch.no_grad():
            loss = self.model(input_ids=input_ids, labels=input_ids).loss

        tokens_num = len(input_ids)
        loss_value = loss.item() * tokens_num
        return loss_value, tokens_num

    def loss_text(self, text):
        total_loss = 0
        total_tokens = 0

        loss_value, token_count = self.calculate_loss(text)
        total_loss = loss_value
        total_tokens = token_count

        ppl = math.exp(total_loss / total_tokens)

        return ppl


    def reply(self, ctx: ChatContext, cands_num: int = 4, max_response_len=None, temperature=None, timeout=MODEL_TIMEOUT):
        logger.debug('=== %s |||  Message: %s' % (datetime.now().strftime("%H:%M:%S"), ctx.message))
        logger.debug(dumps(ctx, default=vars, indent=2))

        sep = self.tokenizer.eos_token
        context = encode_context(ctx) + ['Bot: ']
        context_str = sep.join(context)
        logger.debug(context_str)
        resp = self.reply_raw_gpt(context_str, cands_num, max_response_len, temperature, timeout)
        logger.debug(resp)
        return list(resp.keys())[:cands_num], list(resp.keys())[0] if len(resp) > 0 else 'Sorry, I could not answer'


    def reply_raw_gpt(self, context_str: str, cands_num: int = 4, max_response_len=None, temperature=None, timeout=MODEL_TIMEOUT):
        encoded_context = self.tokenizer.encode(context_str, return_tensors="pt").to(self.acceleration)
        context_len = encoded_context.shape[-1]

        if not max_response_len:
            max_response_len = self.params.generation_length
        if not cands_num:
            cands_num = 4

        resp = {}
        if not temperature:
            temperature = self.params.temperature

        time_start = time.time() * 1000
        cur_time = time.time() * 1000
        logger.debug(f'temp: {temperature}, top_k: {self.params.top_k}, max_length: {context_len + max_response_len}, variants: {cands_num}')
        while len(resp) < cands_num and cur_time - time_start < timeout:
            if len(resp) > 0:
                logger.debug('Giving one more attempt...')

            num_variants_reqd = estimate_candidates(cands_num - len(resp))
            logger.trace('Replies needed: %d, will generate %d candidates' % (cands_num, num_variants_reqd))
            responses_ids = self.model.generate(
                encoded_context,
                use_cache=True,
                do_sample=True,
                max_length=context_len + max_response_len,
                top_p=self.params.top_p,
                top_k=self.params.top_k,
                temperature=temperature,
                num_return_sequences=num_variants_reqd,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            for response in responses_ids:
                spl = [v for v in self.tokenizer.decode(response).strip().split(self.tokenizer.eos_token) if len(v) > 0]
                r_raw = spl[-1]
                r = r_raw.replace(context_str, '').replace(self.tokenizer.eos_token, '').strip()
                option = strip_prefix(r).strip()

                reject = False
                for bl_item in blacklist:
                    if bl_item in option:
                        print('%s is rejected due to blacklist...' % r)
                        reject = True
                ls = self.loss_text(response)
                # r = self.tokenizer.decode(response).replace(context_str,'')                    
                if not reject:
                    resp[option] = ls + r.count('*')
                
            cur_time = time.time() * 1000

        torch.cuda.empty_cache()
        print('Time: %f' % (cur_time - time_start))  
        # sort by descending and return cands_num 
        resp = dict(sorted(resp.items(), key=lambda item: item[1])) 
        return resp if len(resp) > 0 else ['Sorry, I could not answer']


def detect_temperature(context: ChatContext, gpt: GPTNeoWrapper):
    if context.temp:
        return context.temp
    if not context.history:
        return gpt.params.temperature 

    # Implement temperature increase with each turn
    return 1.2 - 1 / (len(context.history) / 2)

