BASE_MODEL_NAME = "gpt2-large"
MODEL_PREFIX = 'models/'
MODEL_NAME = "models/gptneo_2.7b_330k"
# MODEL_NAME = 'models/checkpoint_best_on_330000'
VERSION = '0.0.1demo'   # Friendly 330k
ACCELERATION = 'cuda:0'

MODEL_TIMEOUT = 3000 # ms
DEFAULT_PROFILE = 'My name is {name}. I am {gender}{age}. I am from LA'
MAX_CONTEXT_TURNS = 18

class Args:
    def __init__(self):
        self.overwrite_cache = False
        self.tokenizer_name = BASE_MODEL_NAME
        self.model_name_or_path = BASE_MODEL_NAME
        self.output_dir = "model_dir2"
        self.cache_dir = "cache_dir"
        self.eval_batch_size = 2
        self.train_batch_size = 2
        self.gradient_accumulation_steps = 32
        self.learning_rate = 5e-5
        self.warmup_steps = 0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 1
        self.eval_steps = 10000
        self.seed = 42
        self.fp16 = True
        self.fp16_opt_level = "O1"

class GPTNeoParams:
    temperature = 0.5
    top_p = 0.98
    top_k = 7000
    generation_length = 24
    repetition_thresh = 0.5
    loss_thresh = 39

class MistralParams:
    n = 1
    temperature = 0.8
    top_p = 0.85
    use_beam_search = False
    length_penalty = 1
    early_stopping = False
    max_tokens = 1024
    stop_token_ids = [
        128039
    ]
    ignore_eos = False,
    skip_special_tokens = True,
    spaces_between_special_tokens = True
