import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import GenerativeModel, NERModel
from dataset import GenDataset, EEDataset
from utils import compute_f1
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--e2e_config', required=True)
parser.add_argument('-e', '--e2e_model', required=True)
parser.add_argument('--no_dev', action='store_true', default=False)
parser.add_argument('--ic', action='store_true', default=False)
parser.add_argument('--ner', action='store_true', default=False)
parser.add_argument('--eval_batch_size', type=int)
parser.add_argument('--write_file', type=str)
args = parser.parse_args()
with open(args.e2e_config) as fp:
    config = json.load(fp)
config = Namespace(**config)

if config.dataset == "ace05e" or config.dataset == "ace05ep":
    from template_generate_ace import eve_template_generator
    template_file = "template_generate_ace"
elif config.dataset == "ere":
    from template_generate_ere import eve_template_generator
    template_file = "template_generate_ere"

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def process_events(passage, triggers, roles):
    """
    Given a list of token and event annotation, return a list of structured event

    structured_event:
    {
        'trigger text': str,
        'trigger span': (start, end),
        'event type': EVENT_TYPE(str),
        'arguments':{
            ROLE_TYPE(str):[{
                'argument text': str,
                'argument span': (start, end)
            }],
            ROLE_TYPE(str):...,
            ROLE_TYPE(str):....
        }
        'passage': PASSAGE
    }
    """

    events = {trigger: [] for trigger in triggers}

    for argument in roles:
        trigger = argument[0]
        events[trigger].append(argument)

    event_structures = []
    for trigger, arguments in events.items():
        eve_type = trigger[2]
        eve_text = ' '.join(passage[trigger[0]:trigger[1]])
        eve_span = (trigger[0], trigger[1])
        argus = {}
        for argument in arguments:
            role_type = argument[1][2]
            if role_type not in argus.keys():
                argus[role_type] = []
            argus[role_type].append({
                'argument text': ' '.join(passage[argument[1][0]:argument[1][1]]),
                'argument span': (argument[1][0], argument[1][1]),
            })
        event_structures.append({
            'trigger text': eve_text,
            'trigger span': eve_span,
            'event type': eve_type,
            'arguments': argus,
            'passage': ' '.join(passage),
            'tokens': passage
        })

    return event_structures

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# logger
log_path = os.path.join(os.path.dirname(args.e2e_model), "eval_keywords.log")
logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', force=True,
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")

# set GPU device
torch.cuda.set_device(config.gpu_device)

# check valid styles
assert np.all([style in ['event_type_sent', 'ner_keywords', 'keywords', 'template'] for style in config.input_style])
assert np.all([style in ['trigger:sentence', 'argument:sentence'] for style in config.output_style])
              
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
# special_tokens = ['<Trigger>', '<sep>']
special_tokens = ['<Trigger>', '<sep>', '<and>', '<Keyword>', '</Keyword>']
tokenizer.add_tokens(special_tokens)

if args.eval_batch_size:
    config.eval_batch_size=args.eval_batch_size

ner_tokenizer = None
ner_model = None
mapping = None
if args.ner:
    with open(os.path.join(os.path.dirname(config.train_file), "etypes.json"), 'r') as f:
        mapping = {"id_to_keyword": {}, "keyword_to_id": {}}
        infos = json.load(f)
        for i, label in enumerate(infos['Keyword_type']):
            mapping["id_to_keyword"][i] = label
            mapping["keyword_to_id"][label] = i
    ner_tokenizer = AutoTokenizer.from_pretrained(config.keyword_model_name, cache_dir=config.cache_dir)
    ner_model = NERModel(config, len(mapping['keyword_to_id']), ner_tokenizer)
    ner_model.load_state_dict(torch.load(config.keyword_model))
    ner_model.cuda(config.gpu_device)
    ner_model.eval()

# load data
dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)
with open(config.vocab_file) as f:
    vocab = json.load(f)

# load model
logger.info(f"Loading model from {args.e2e_model}")
model = GenerativeModel(config, tokenizer)
model.load_state_dict(torch.load(args.e2e_model, map_location=f'cuda:{config.gpu_device}'))
model.cuda(device=config.gpu_device)
model.eval()

keyword_write_output = []
dev_gold_key_num, dev_pred_key_num, dev_match_key_num = 0, 0, 0
# eval dev set
if not args.no_dev:
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
    dev_gold_triggers, dev_gold_roles, dev_pred_triggers, dev_pred_roles = [], [], [], []
    
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        
        gold_events = []
        p_triggers = [[] for _ in range(len(batch.tokens))]
        p_roles = [[] for _ in range(len(batch.tokens))]
        for event_type in vocab['event_type_itos']:
            theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)
            
            origin_inputs = []
            for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
                template = theclass(config.input_style, config.output_style, tokens, event_type)
                origin_inputs.append(template.generate_keywords_input_str())
                gold_events.append(process_events(tokens, triggers, roles))
            
            inputs = tokenizer(origin_inputs, return_tensors='pt', padding=True, max_length=config.max_length)
            enc_idxs = inputs['input_ids'].cuda()
            enc_attn = inputs['attention_mask'].cuda()
            
            outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, num_beams=config.beam_size, max_length=config.max_output_length)
            final_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
            
            for bid, (tokens, p_text) in enumerate(zip(batch.tokens, final_outputs)):
                template = theclass(config.input_style, config.output_style, tokens, event_type, gold_event=gold_events[bid])
                pred_object = template.decode_keywords(p_text)
                sub_scores = template.evaluate_keywords(pred_object)

                dev_gold_key_num += sub_scores['gold_num']
                dev_pred_key_num += sub_scores['pred_num']
                dev_match_key_num += sub_scores['match_num']
                keyword_write_output.append({
                    'input text': origin_inputs[bid],
                    'gold text': template.generate_keywords_output_str()[0],
                    'pred text': p_text,
                    'gold keyword spans': template.get_keyword_spans(),
                    'pred keyword spans': pred_object,
                    'score': sub_scores,
                    # 'gold info': keyword_info
                })

                
    progress.close()
    dev_scores = {
        'keyword_cls': compute_f1(dev_pred_key_num, dev_gold_key_num, dev_match_key_num),
    }

    logger.info("---------------------------------------------------------------------")
    logger.info('Keyword C  - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
        dev_scores['keyword_cls'][0] * 100.0, dev_match_key_num, dev_pred_key_num, 
        dev_scores['keyword_cls'][1] * 100.0, dev_match_key_num, dev_gold_key_num, dev_scores['keyword_cls'][2] * 100.0))
    logger.info("---------------------------------------------------------------------")
    
    
# test set
keyword_write_output = []
test_gold_key_num, test_pred_key_num, test_match_key_num = 0, 0, 0
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)

    gold_events = []
    p_texts = [[] for _ in range(len(batch.tokens))]
    g_texts = [[] for _ in range(len(batch.tokens))]
    p_keyword_spans = [[] for _ in range(len(batch.tokens))]
    g_keyword_spans = [[] for _ in range(len(batch.tokens))]
    for event_type in vocab['event_type_itos']:
        theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)
        
        origin_inputs = []
        for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
            template = theclass(config.input_style, config.output_style, tokens, event_type)
            origin_inputs.append(template.generate_keywords_input_str())
            gold_events.append(process_events(tokens, triggers, roles))
        
        inputs = tokenizer(origin_inputs, return_tensors='pt', padding=True, max_length=config.max_length)
        enc_idxs = inputs['input_ids'].cuda()
        enc_attn = inputs['attention_mask'].cuda()
        
        outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, num_beams=config.beam_size, max_length=config.max_output_length)
        final_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
        
        for bid, (tokens, p_text) in enumerate(zip(batch.tokens, final_outputs)):
            template = theclass(config.input_style, config.output_style, tokens, event_type, gold_event=gold_events[bid])
            pred_object = template.decode_keywords(p_text)

            sub_scores = template.evaluate_keywords(pred_object)
            test_gold_key_num += sub_scores['gold_num']
            test_pred_key_num += sub_scores['pred_num']
            test_match_key_num += sub_scores['match_num']
            p_texts[bid].append(p_text)
            g_texts[bid].append(template.generate_keywords_output_str()[0])
            p_keyword_spans[bid].append(pred_object)
            g_keyword_spans[bid].append(template.get_keyword_spans())

    for tokens, pt, gt, pks, gks in zip(batch.tokens, p_texts, g_texts, p_keyword_spans, g_keyword_spans):
        keyword_write_output.append({
            'input text': " ".join(tokens),
            'gold text': gt,
            'pred text': pt,
            'gold keyword spans': gks,
            'pred keyword spans': pks,
        })
            
            
progress.close()

test_scores = {
    'keyword_cls': compute_f1(test_pred_key_num, test_gold_key_num, test_match_key_num),
}

logger.info("---------------------------------------------------------------------")
logger.info('Keyword C  - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['keyword_cls'][0] * 100.0, test_match_key_num, test_pred_key_num, 
    test_scores['keyword_cls'][1] * 100.0, test_match_key_num, test_gold_key_num, test_scores['keyword_cls'][2] * 100.0))
logger.info("---------------------------------------------------------------------")

if args.write_file:
    with open(args.write_file, 'w') as fw:
        json.dump(keyword_write_output, fw, indent=4)
