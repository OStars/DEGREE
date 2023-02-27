import os, sys, json, logging, time, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model import NERModel
from dataset import NERDataset
from utils import Summarizer, compute_f1, convert_tag_sequence_to_span
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
# set GPU device
torch.backends.cudnn.enabled = False

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', force=True,
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

torch.cuda.set_device(config.gpu_device)

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

with open(os.path.join(config.data_dir, "etypes.json"), 'r') as f:
    mapping = {
        "id_to_keyword": {}, 
        "keyword_to_id": {},
        "id_to_event": {},
        "event_to_id": {}
    }
    infos = json.load(f)
    for i, label in enumerate(infos['Keyword_type']):
        mapping["id_to_keyword"][i] = label
        mapping["keyword_to_id"][label] = i
    for i, label in enumerate(infos['Event_type']):
        mapping["id_to_event"][i] = label
        mapping["event_to_id"][label] = i

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.keyword_model_name, cache_dir=config.cache_dir)

train_set = NERDataset(tokenizer, config.train_file, mapping, config.n_negative, config.max_length)
dev_set = NERDataset(tokenizer, config.dev_file, mapping, config.n_negative, config.max_length)
test_set = NERDataset(tokenizer, config.test_file, mapping, config.n_negative, config.max_length)
train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

model = NERModel(config, len(mapping['id_to_keyword']), tokenizer)
model.cuda(device=config.gpu_device)

# optimizer
param_groups = [{
    'params': filter(lambda p: p.requires_grad, model.parameters()),
    'lr': config.learning_rate,
    'weight_decay': config.weight_decay
}]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num*config.warmup_epoch,
                                           num_training_steps=train_batch_num*config.max_epoch)

# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'precision': 0.0,
    'recall': 0.0,
    'f1': 0.0
}

for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")

    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step, 
                                                    shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        loss = model(batch)
        loss = torch.sum(loss) / len(loss)

        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1
        
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()

    # eval dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
    model.eval()
    best_dev_flag = False
    write_output = []

    dev_gold_key_num, dev_pred_key_num, dev_match_key_num = 0, 0, 0
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size, 
                                                 shuffle=False, collate_fn=dev_set.collate_fn)):
        progress.update(1)
        prediction = model.predict(batch)

        gold_keyword_label_texts = [
            [mapping['id_to_keyword'][label] for label in keyword_labels if label != -1] 
            for keyword_labels in batch.keyword_labels.detach().numpy()
        ]
        pred_keyword_label_texts = [
            [mapping['id_to_keyword'][label] for label in keyword_labels if label != -1] 
            for keyword_labels in prediction
        ]
        gold_keywords = [convert_tag_sequence_to_span(keywords) for keywords in gold_keyword_label_texts]
        pred_keywords = [convert_tag_sequence_to_span(keywords) for keywords in pred_keyword_label_texts]
        for g_keywords, p_keywords, tokens, wnd_id in zip(gold_keywords, pred_keywords, batch.tokens, batch.wnd_ids):
            dev_gold_key_num += len(g_keywords)
            dev_pred_key_num += len(p_keywords)
            for p_keyword in p_keywords:
                if p_keyword in g_keywords:
                    dev_match_key_num += 1
            
            g_keywords_text = [' '.join(tokens[start:end]) for start, end in g_keywords]
            p_keywords_text = [' '.join(tokens[start:end]) for start, end in p_keywords]
            write_output.append({
                'wnd_id': wnd_id,
                'input text': ' '.join(tokens),
                'gold keywords': g_keywords_text,
                'pred keywords': p_keywords_text,
                'gold keywords indices': g_keywords,
                'pred keywords indices': p_keywords,
            })
    progress.close()

    precision, recall, f1 = compute_f1(dev_pred_key_num, dev_gold_key_num, dev_match_key_num)
    dev_scores = {'precision': precision, 'recall': recall, 'f1': f1}

    # print scores
    logger.info("---------------------------------------------------------------------")
    logger.info('Result  - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
        dev_scores['precision'] * 100.0, dev_match_key_num, dev_pred_key_num, 
        dev_scores['recall'] * 100.0, dev_match_key_num, dev_gold_key_num, dev_scores['f1'] * 100.0))
    logger.info("---------------------------------------------------------------------")

    # check best dev model
    if dev_scores['f1'] > best_dev_scores['f1']:
        best_dev_flag = True

    if best_dev_flag:    
        best_dev_scores = dev_scores
        best_dev_epoch = epoch
        
        # save best model
        logger.info('Saving best model')
        torch.save(model.state_dict(), best_model_path)
        
        # save dev result
        with open(dev_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)

        # eval test set
        progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
        write_output = []

        test_gold_key_num, test_pred_key_num, test_match_key_num = 0, 0, 0
        for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size, 
                                                     shuffle=False, collate_fn=test_set.collate_fn)):
            progress.update(1)
            prediction = model.predict(batch)

            gold_keyword_label_texts = [
                [mapping['id_to_keyword'][label] for label in keyword_labels if label != -1]
                for keyword_labels in batch.keyword_labels.detach().numpy()
            ]
            pred_keyword_label_texts = [
                [mapping['id_to_keyword'][label] for label in keyword_labels if label != -1]
                for keyword_labels in prediction
            ]
            gold_keywords = [convert_tag_sequence_to_span(keywords) for keywords in gold_keyword_label_texts]
            pred_keywords = [convert_tag_sequence_to_span(keywords) for keywords in pred_keyword_label_texts]
            for g_keywords, p_keywords, tokens, wnd_id in zip(gold_keywords, pred_keywords, batch.tokens, batch.wnd_ids):
                test_gold_key_num += len(g_keywords)
                test_pred_key_num += len(p_keywords)
                for p_keyword in p_keywords:
                    if p_keyword in g_keywords:
                        test_match_key_num += 1
                
                g_keywords_text = [' '.join(tokens[start:end]) for start, end in g_keywords]
                p_keywords_text = [' '.join(tokens[start:end]) for start, end in p_keywords]
                write_output.append({
                    'wnd_id': wnd_id,
                    'input text': ' '.join(tokens),
                    'gold keywords': g_keywords_text,
                    'pred keywords': p_keywords_text,
                    'gold keywords indices': g_keywords,
                    'pred keywords indices': p_keywords,
                })
        progress.close()

        precision, recall, f1 = compute_f1(dev_pred_key_num, dev_gold_key_num, dev_match_key_num)
        test_scores = {'precision': precision, 'recall': recall, 'f1': f1}
        # print scores
        logger.info("---------------------------------------------------------------------")
        logger.info('Result  - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
            test_scores['precision'] * 100.0, test_match_key_num, test_pred_key_num, 
            test_scores['recall'] * 100.0, test_match_key_num, test_gold_key_num, test_scores['f1'] * 100.0))
        logger.info("---------------------------------------------------------------------")
        
        # save test result
        with open(test_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)
    
    logger.info({"epoch": epoch, "dev_scores": dev_scores})
    if best_dev_flag:
        logger.info({"epoch": epoch, "test_scores": test_scores})
    logger.info("Current best")
    logger.info({"best_epoch": best_dev_epoch, "best_scores": best_dev_scores})
        
logger.info(log_path)
logger.info("Done!")