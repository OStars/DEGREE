import torch
import json, logging, pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import namedtuple
from utils import pad_sequence_to_length

logger = logging.getLogger(__name__)

ee_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'triggers', 'roles']
ee_batch_fields = ['tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'triggers', 'roles', 'wnd_ids']
EEInstance = namedtuple('EEInstance', field_names=ee_instance_fields, defaults=[None] * len(ee_instance_fields))
EEBatch = namedtuple('EEBatch', field_names=ee_batch_fields, defaults=[None] * len(ee_batch_fields))

ner_instance_fields = ['doc_id', 'wnd_id', 'text', 'tokens', 'pieces', 'piece_idxs', 'keyword_labels', 'offsets']
ner_batch_fields = ['input_ids', 'attention_mask', 'token_type_ids', 'mask', 'offsets', 'tokens', 'pieces', 'piece_idxs', 'keyword_labels', 'wnd_ids']
NERInstance = namedtuple('NERInstance', field_names=ner_instance_fields, defaults=[None] * len(ner_instance_fields))
NERBatch = namedtuple('NERBatch', field_names=ner_batch_fields, defaults=[None] * len(ner_batch_fields))

gen_batch_fields = ['input_text', 'target_text', 'enc_idxs', 'enc_attn', 'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs', 'infos']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        break_flag = False
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                break_flag = True
        if break_flag:
            continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map

def get_role_list(entities, events, id_map):
    entity_idxs = {entity['id']: (i,entity) for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(arg['entity_id'], arg['entity_id'])]
            
            # This will automatically remove multi role scenario
            if visited[i][entity_idx[0]] == 0:
                # ((trigger start, trigger end, trigger type), (argument start, argument end, role type))
                temp = ((event['trigger']['start'], event['trigger']['end'], event['event_type']),
                        (entity_idx[1]['start'], entity_idx[1]['end'], arg['role']))
                role_list.append(temp)
                visited[i][entity_idx[0]] = 1
    role_list.sort(key=lambda x: (x[0][0], x[1][0]))
    return role_list

class EEDataset(Dataset):
    def __init__(self, tokenizer, path, max_length=128, fair_compare=True):
        self.tokenizer = tokenizer
        self.path = path
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.fair_compare = fair_compare
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_length:
                continue
            self.insts.append(inst)

        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            
            entities = inst['entity_mentions']
            if self.fair_compare:
                entities, entity_id_map = remove_overlap_entities(entities)
            else:
                entities = entities
                entity_id_map = {}
                
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
            assert sum(token_lens) == len(piece_idxs)
                        
            triggers = [(e['trigger']['start'], e['trigger']['end'], e['event_type']) for e in events]
            no_duplicated_triggers = set(triggers)
            assert len(triggers) == len(no_duplicated_triggers)
            roles = get_role_list(entities, events, entity_id_map)
            
            token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]
            
            instance = EEInstance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                token_start_idxs=token_start_idxs,
                triggers = triggers,
                roles = roles,
            )
            self.data.append(instance)
            
        logger.info(f'Loaded {len(self)}/{len(lines)} instances from {self.path}')

    def collate_fn(self, batch):
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        piece_idxs = [inst.piece_idxs for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        token_start_idxs = [inst.token_start_idxs for inst in batch]
        triggers = [inst.triggers for inst in batch]
        roles = [inst.roles for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]

        return EEBatch(
            tokens=tokens,
            pieces=pieces,
            piece_idxs=piece_idxs,
            token_lens=token_lens,
            token_start_idxs=token_start_idxs,
            triggers=triggers,
            roles=roles,
            wnd_ids=wnd_ids,
        )

class GenDataset(Dataset):
    def __init__(self, tokenizer, max_length, path, max_output_length=None, unseen_types=[], no_bos=False):
        self.tokenizer = tokenizer
        self.max_length = self.max_output_length = max_length
        if max_output_length is not None:
            self.max_output_length = max_output_length
        self.path = path
        self.no_bos = no_bos # if you use bart, then this should be False; if you use t5, then this should be True
        self.data = []
        self.load_data(unseen_types)
        # self.data = self.data[:100] # FOR DEBUG

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, unseen_types):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        for l_in, l_out, l_info in zip(data['input'], data['target'], data['all']):
            if len(unseen_types) > 0:
                if isinstance(l_info, tuple):
                    # instance base
                    if l_info[1] in unseen_types:
                        continue
                else:
                    # trigger base, used in argument model
                    if l_info['event type'] in unseen_types:
                        continue
            self.data.append({
                'input': l_in,
                'target': l_out,
                'info': l_info
            })
        logger.info(f'Loaded {len(self)} instances from {self.path}')

    def collate_fn(self, batch):
        input_text = [x['input'] for x in batch]
        target_text = [x['target'] for x in batch]
        
        # encoder inputs
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.max_length)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(target_text, return_tensors='pt', padding=True, max_length=self.max_output_length)
        dec_idxs = targets['input_ids']
        batch_size = dec_idxs.size(0)
        dec_idxs[:, 0] = self.tokenizer.eos_token_id
        dec_attn = targets['attention_mask']
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return GenBatch(
            input_text=input_text,
            target_text=target_text,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            raw_lbl_idxs=raw_lbl_idxs,
            infos=[x['info'] for x in batch]
        )


class NERDataset(Dataset):
    def __init__(self, tokenizer, path, mapping, n_negative, max_length=128, fair_compare=True):
        self.tokenizer = tokenizer
        self.path = path
        self.data = []
        self.insts = []
        self.mapping = mapping
        self.n_negative = n_negative
        self.max_length = max_length
        self.fair_compare = fair_compare
        self.event_types = self.mapping['event_to_id'].keys()
        self.keyword_types = self.mapping['keyword_to_id'].keys()
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def event_type_set(self):
        if not self.event_types:
            type_set = set()
            for inst in self.insts:
                for event in inst['event_mentions']:
                    type_set.add(event['event_type'])
            self.event_types = type_set
        return self.event_types

    @property
    def keyword_type_set(self):
        if not self.keyword_types:
            type_set = set()
            for d in self.data:
                type_set.update(set(d['keyword_labels']))
            self.keyword_types = type_set
        return self.keyword_types

    def checkLabel(self, labels, start, end):
        for i in range(end - start):
            if labels[start + i] != 'O':
                return False
        return True
    
    def get_keyword_labels(self, event_type, inst):
        prompt = "Extract keywords for " + event_type + " event . "
        prompt_len = len(prompt.split())
        text = prompt + " ".join(inst['tokens'])
        tokens = text.split()

        events = inst['event_mentions']
        events.sort(key=lambda x: x['trigger']['start'])                
        entities = inst['entity_mentions']
        if self.fair_compare:
            entities, entity_id_map = remove_overlap_entities(entities)
        else:
            entities = entities
            entity_id_map = {}
        entities = {entity['id']: entity for entity in entities}

        keywords = []
        keyword_labels = ['O'] * len(tokens)
        for event in events:
            if event['event_type'] == event_type:
                keywords.append((event['trigger']['start'], event['trigger']['end']))
                for arg in event['arguments']:
                    entity_id = arg['entity_id']
                    keywords.append((entities[entity_id]['start'], entities[entity_id]['end']))
        for keyword in keywords:
            start = keyword[0] + prompt_len
            end = keyword[1] + prompt_len
            if self.checkLabel(keyword_labels, start, end):
                keyword_labels[start] = 'B-keyword'
                if end - start > 1:
                    for i in range(end-start-1):
                        keyword_labels[start+i+1] = 'I-keyword'
        
        return text, tokens, keyword_labels

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_length:
                continue
            self.insts.append(inst)

        for inst in tqdm(self.insts):
            pos_types = set([event['event_type'] for event in inst['event_mentions']])
            neg_types = set([event_type for event_type in self.event_types if event_type not in pos_types][:self.n_negative])
            event_types = set()
            event_types.update(pos_types)
            event_types.update(neg_types)
            for event_type in event_types:
                wnd_id = inst['wnd_id']
                text, tokens, keyword_labels = self.get_keyword_labels(event_type=event_type, inst=inst)

                pieces = [self.tokenizer.tokenize(t) for t in tokens]
                token_lens = [len(p) for p in pieces]
                pieces = [p for ps in pieces for p in ps]
                piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
                assert sum(token_lens) == len(piece_idxs)
                
                token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]
                offsets = [(token_start_idxs[i], token_start_idxs[i+1]) for i in range(len(token_start_idxs) - 1)]

                instance = NERInstance(
                    wnd_id=wnd_id,
                    text=text,
                    tokens=tokens,
                    pieces=pieces,
                    piece_idxs=piece_idxs,
                    keyword_labels=keyword_labels,
                    offsets=offsets
                )
                self.data.append(instance)
            
        logger.info(f'Loaded {len(self)}/{len(lines)} instances from {self.path}')

    def collate_fn(self, batch):
        texts = [inst.text for inst in batch]
        tokenized_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        input_ids = tokenized_texts.input_ids
        token_type_ids = tokenized_texts.token_type_ids
        attention_mask = tokenized_texts.attention_mask
        offsets = [inst.offsets for inst in batch]
        keyword_labels = [[self.mapping['keyword_to_id'][keyword] for keyword in inst.keyword_labels] for inst in batch]

        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        piece_idxs = [inst.piece_idxs for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]

        padding_length = max([len(_) for _ in tokens])
        offsets = torch.LongTensor(
            [pad_sequence_to_length(offset, padding_length, default_value=lambda: (0, 0)) for offset in offsets]
        )
        keyword_labels = torch.LongTensor(
            [pad_sequence_to_length(labels, padding_length, lambda: -1) for labels in keyword_labels]
        )
        mask = keyword_labels != -1

        return NERBatch(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mask=mask,
            offsets=offsets,
            keyword_labels=keyword_labels,
            tokens=tokens,
            pieces=pieces,
            piece_idxs=piece_idxs,
            wnd_ids=wnd_ids,
        )