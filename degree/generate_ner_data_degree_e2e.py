import os, json, pickle, logging, pprint, random, torch
import numpy as np
from tqdm import tqdm
from dataset import EEDataset
from argparse import ArgumentParser, Namespace
from utils import generate_vocabs
from transformers import AutoTokenizer
from model import NERModel
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

if config.dataset == "ace05e" or config.dataset == "ace05ep":
    from template_generate_ace import eve_template_generator
elif config.dataset == "ere":
    from template_generate_ere import eve_template_generator

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")

def generate_data(data_set, vocab, config, tokenizer=None, is_train=False, keyword_tokenizer=None, keyword_model=None, mapping=None):
    inputs = []
    targets = []
    events = []
    
    keyword_inputs = []
    keyword_targets = []
    keywords = []

    def organize_data(data, config, is_train=is_train):
        inputs = []
        targets = []
        infos = []

        pos_data_ = [dt for dt in data if dt[3]]
        neg_data_ = [dt for dt in data if not dt[3]]
        np.random.shuffle(neg_data_)
        
        # data => (input_str, output_str, self.gold_event, gold_sample, self.event_type, self.tokens)
        for data_ in pos_data_:
            inputs.append(data_[0])
            targets.append(data_[1])
            if tokenizer is not None:
                infos.append((data_[2], data_[4], data_[5], data_[6], data_[7], data_[8]))
            else:
                infos.append((data_[2], data_[4], data_[5]))
        
        neg_data_ = neg_data_[:config.n_negative]
        # neg_data_ = neg_data_[:config.n_negative] if is_train else neg_data_
        for data_ in neg_data_:
            inputs.append(data_[0])
            targets.append(data_[1])
            if tokenizer is not None:
                infos.append((data_[2], data_[4], data_[5], data_[6], data_[7], data_[8]))
            else:
                infos.append((data_[2], data_[4], data_[5]))
        
        return inputs, targets, infos

    for data in tqdm(data_set.data):
        event_template = eve_template_generator(data.tokens, data.triggers, data.roles, config.input_style, config.output_style, vocab, True, keyword_tokenizer=keyword_tokenizer, keyword_model=keyword_model, mapping=mapping)
        # event_template = eve_template_generator(data.tokens, data.triggers, data.roles, config.input_style, config.output_style, vocab, True, tokenizer=tokenizer)
        # event_template = eve_template_generator(data.tokens, data.triggers, data.roles, config.input_style, config.output_style, vocab, True, is_train=is_train)
        # if event_template.events:
        event_data, keyword_data = event_template.get_training_data()
        inputs_, targets_, events_ = organize_data(event_data, config)
        inputs.extend(inputs_)
        targets.extend(targets_)
        events.extend(events_)

        # inputs_, targets_, keywords_ = organize_data(keyword_data, config)
        # keyword_inputs.extend(inputs_)
        # keyword_targets.extend(targets_)
        # keywords.extend(keywords_)
    
    return inputs, targets, events
    # return inputs, targets, events, keyword_inputs, keyword_targets, keywords

# check valid styles
assert np.all([style in ['event_type_sent', 'keywords', 'ner_keywords', 'template'] for style in config.input_style])
assert np.all([style in ['keywords_chain', 'trigger:sentence', 'argument:sentence'] for style in config.output_style])

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
special_tokens = ['<Trigger>', '<sep>', '<and>',  '<Keyword>', '</Keyword>']
tokenizer.add_tokens(special_tokens)

if not os.path.exists(config.finetune_dir):
    os.makedirs(config.finetune_dir)

# load data
train_set = EEDataset(tokenizer, config.train_file, max_length=config.max_length)
dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
vocab = generate_vocabs([train_set, dev_set, test_set])

# save vocabulary
with open('{}/vocab.json'.format(config.finetune_dir), 'w') as f:
    json.dump(vocab, f, indent=4)    


with open(os.path.join(os.path.dirname(config.train_file), "etypes.json"), 'r') as f:
    mapping = {"id_to_keyword": {}, "keyword_to_id": {}}
    infos = json.load(f)
    for i, label in enumerate(infos['Keyword_type']):
        mapping["id_to_keyword"][i] = label
        mapping["keyword_to_id"][label] = i

# Model for keyword extraction
ner_tokenizer = AutoTokenizer.from_pretrained(config.keyword_model_name, cache_dir=config.cache_dir)
model = NERModel(config, len(mapping['keyword_to_id']), ner_tokenizer)
model.load_state_dict(torch.load(config.keyword_model))
model.cuda(config.gpu_device)
model.eval()

# generate finetune data
train_inputs, train_targets, train_events = generate_data(train_set, vocab, config, keyword_tokenizer=ner_tokenizer, keyword_model=model, is_train=True, mapping=mapping)
logger.info(f"Generated {len(train_inputs)} training examples from {len(train_set)} instance")

with open('{}/train_input.json'.format(config.finetune_dir), 'w') as f:
    json.dump(train_inputs, f, indent=4)

with open('{}/train_target.json'.format(config.finetune_dir), 'w') as f:
    json.dump(train_targets, f, indent=4)

with open('{}/train_all.pkl'.format(config.finetune_dir), 'wb') as f:
    pickle.dump({
        'input': train_inputs,
        'target': train_targets,
        'all': train_events
    }, f)

    
dev_inputs, dev_targets, dev_events = generate_data(dev_set, vocab, config, keyword_tokenizer=ner_tokenizer, keyword_model=model, mapping=mapping)
logger.info(f"Generated {len(dev_inputs)} dev examples from {len(dev_set)} instance")

with open('{}/dev_input.json'.format(config.finetune_dir), 'w') as f:
    json.dump(dev_inputs, f, indent=4)

with open('{}/dev_target.json'.format(config.finetune_dir), 'w') as f:
    json.dump(dev_targets, f, indent=4)

with open('{}/dev_all.pkl'.format(config.finetune_dir), 'wb') as f:
    pickle.dump({
        'input': dev_inputs,
        'target': dev_targets,
        'all': dev_events
    }, f)


test_inputs, test_targets, test_events = generate_data(test_set, vocab, config, keyword_tokenizer=ner_tokenizer, keyword_model=model, mapping=mapping)
logger.info(f"Generated {len(test_inputs)} test examples from {len(test_set)} instance")

with open('{}/test_input.json'.format(config.finetune_dir), 'w') as f:
    json.dump(test_inputs, f, indent=4)

with open('{}/test_target.json'.format(config.finetune_dir), 'w') as f:
    json.dump(test_targets, f, indent=4)

with open('{}/test_all.pkl'.format(config.finetune_dir), 'wb') as f:
    pickle.dump({
        'input': test_inputs,
        'target': test_targets,
        'all': test_events
    }, f)