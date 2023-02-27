import logging
import numpy as np
import torch
import torch.nn as nn
from TorchCRF import CRF
from utils import batched_span_select
from transformers import AutoConfig, AutoModelForPreTraining, AutoModel
import ipdb

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        logger.info(f'Loading pre-trained model {config.model_name}')
        self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        self.model = AutoModelForPreTraining.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs, 
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=4, max_length=50):
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=batch.enc_idxs, 
                                          attention_mask=batch.enc_attn, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
            
        final_output = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()

        return final_output


class NERModel(nn.Module):
    def __init__(self, config, num_class, tokenizer):
        super().__init__()
        self.device = config.gpu_device
        self.tokenizer = tokenizer
        self.num_class = num_class
        logger.info(f'Loading pre-trained model {config.keyword_model_name}')
        self.model_config =  AutoConfig.from_pretrained(config.keyword_model_name, cache_dir=config.cache_dir)
        self.model = AutoModel.from_pretrained(config.keyword_model_name, cache_dir=config.cache_dir, config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.dropout = nn.Dropout(config.dropout)
        # self.birnn = nn.LSTM(self.model_config.hidden_size, self.model_config.hidden_size // 2, 
        #                      num_layers=1, bidirectional=True, batch_first=True)

        self.emission_layer = nn.Linear(self.model_config.hidden_size, self.num_class)
        self.crf = CRF(self.num_class)

    def forward(self, batch):
        embeddings = self.get_bert_embedding(batch)

        # embeddings, _ = self.birnn(embeddings)
        emissions = self.emission_layer(self.dropout(embeddings))
        
        mask = batch.mask.to(self.device)
        keyword_labels = batch.keyword_labels.to(self.device)
        log_likelihood = -1 * self.crf(emissions, keyword_labels, mask=mask)
        
        return log_likelihood
    
    def predict(self, batch):
        embeddings = self.get_bert_embedding(batch)

        # embeddings, _ = self.birnn(embeddings)
        emissions = self.emission_layer(self.dropout(embeddings))
        
        mask = batch.mask.to(self.device)
        sequence_of_tags = self.crf.viterbi_decode(emissions, mask=mask)

        return sequence_of_tags
    
    def get_bert_embedding(self, batch):
        outputs = self.model(input_ids=batch.input_ids.to(self.device), 
                            attention_mask=batch.attention_mask.to(self.device),
                            token_type_ids=batch.token_type_ids.to(self.device))
        embeddings = outputs.last_hidden_state
        
        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = batched_span_select(embeddings.contiguous(), batch.offsets.to(self.device))
        span_mask = span_mask.unsqueeze(-1)
        # Shape: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        span_embeddings *= span_mask  # zero out paddings
        # Sum over embeddings of all sub-tokens of a word
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        span_embeddings_sum = span_embeddings.sum(2)
        # Shape (batch_size, num_orig_tokens)
        span_embeddings_len = span_mask.sum(2)
        # Find the average of sub-tokens embeddings by dividing `span_embedding_sum` by `span_embedding_len`
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)
        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        return orig_embeddings