'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import transformers
transformers.logging.set_verbosity_error()

from models.fflip import (
    VisionConfig, 
    VisionModel,
    BertModel, 
    BertConfig,
    Attention,
    init_tokenizer,
    load_checkpoint)

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class fs2ksde_fflip_refine(nn.Module):
    def __init__(self,                 
                 text_config = '/home/ubuntu/workplace/fsy/FVLP/configs/bert_config.json', 
                 visual_config = '/home/ubuntu/workplace/fsy/FVLP/configs/vision_config.json',
                 vit = 'base',
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 ):
                  
        super().__init__()
       
        self.vision_config = VisionConfig().from_json_file(visual_config)
        self.visual_encoder = VisionModel.from_pretrained("openai/clip-vit-base-patch16", config = self.vision_config)
        vision_width = self.visual_encoder.config.hidden_size

        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(text_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased',config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(2048, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders
        self.visual_encoder_m = VisionModel(config = self.vision_config)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)
        self.vision_proj_m = nn.Linear(2048, embed_dim)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("sketch_sde_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_sketch_sde_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.sketch_sde_queue = nn.functional.normalize(self.sketch_sde_queue, dim=0)
        self.text_sketch_sde_queue = nn.functional.normalize(self.text_sketch_sde_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.negative_all_rank = negative_all_rank
        self.conv = nn.Conv2d(in_channels=768*4, out_channels=2048, kernel_size=3, stride=2)
        self.attn = Attention()

        nn.init.xavier_uniform_(self.vision_proj.weight)
        nn.init.xavier_uniform_(self.vision_proj_m.weight)
        
    def forward(self, image, sketch_sde, caption, alpha, idx):
        self.train()
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        # image
        image_embeds = self.visual_encoder(image, intermediate_hidden_state = True)
        image_atts = torch.ones(image_embeds.last_hidden_state.size()[:-1], dtype=torch.long).to(image.device)
        layer_embeds = attention(self,image_embeds).to(image.device)
    
        
        # sketch
        sketch_sde_embeds = self.visual_encoder(sketch_sde, intermediate_hidden_state = True)
        sketch_sde_atts = torch.ones(sketch_sde_embeds.last_hidden_state.size()[:-1], dtype=torch.long).to(sketch_sde.device)
        s_layer_embeds = attention(self, sketch_sde_embeds).to(sketch_sde.device)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=65,return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        image_feat = F.normalize(self.vision_proj(layer_embeds), dim=-1)
        sketch_sde_feat = F.normalize(self.vision_proj(s_layer_embeds), dim=-1)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        text_sketch_sde_feat = F.normalize(self.vision_proj(s_layer_embeds) + self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=1)

        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image, intermediate_hidden_state = True)
            layer_embeds_m = attention(self, image_embeds_m).to(image.device)
            image_feat_m = F.normalize(self.vision_proj_m(layer_embeds_m), dim=-1)
            image_feat_m_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            sketch_sde_embeds_m = self.visual_encoder_m(sketch_sde, intermediate_hidden_state = True)
            s_layer_embeds_m = attention(self, sketch_sde_embeds_m).to(sketch_sde.device)
            sketch_sde_feat_m = F.normalize(self.vision_proj_m(s_layer_embeds_m), dim=-1)
            sketch_sde_feat_m_all = torch.cat([sketch_sde_feat_m.t(), self.sketch_sde_queue.clone().detach()], dim=1)
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_m_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            text_sketch_sde_feat_m = F.normalize(self.vision_proj_m(s_layer_embeds_m) + self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=1)

            text_sketch_sde_feat_m_all = torch.cat([text_sketch_sde_feat_m.t(), self.text_sketch_sde_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp
            sim_s2i_m = sketch_sde_feat_m @ image_feat_m_all / self.temp
            sim_i2s_m = image_feat_m @ sketch_sde_feat_m_all / self.temp
            sim_ts2i_m = text_sketch_sde_feat_m @ image_feat_m_all / self.temp
            sim_i2ts_m = image_feat_m @ text_sketch_sde_feat_m_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_s2i_targets = alpha * F.softmax(sim_s2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_i2s_targets = alpha * F.softmax(sim_i2s_m, dim=1) + (1 - alpha) * sim_targets
            sim_ts2i_targets = alpha * F.softmax(sim_ts2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_i2ts_targets = alpha * F.softmax(sim_i2ts_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ image_feat_m_all / self.temp
        sim_s2i = sketch_sde_feat @ image_feat_m_all / self.temp
        sim_i2s = image_feat @ sketch_sde_feat_m_all / self.temp
        sim_ts2i = text_sketch_sde_feat @ image_feat_m_all / self.temp
        sim_i2ts = image_feat @ text_sketch_sde_feat_m_all / self.temp
        
        # image-text
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        # sketch image-text
        loss_ts2i = -torch.sum(F.log_softmax(sim_ts2i, dim=1) * sim_ts2i_targets, dim=1).mean()
        loss_i2ts = -torch.sum(F.log_softmax(sim_i2ts, dim=1) * sim_i2ts_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i + loss_ts2i + loss_i2ts) / 4

        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue_flip(image_feat_m, text_feat_m, sketch_sde_feat_m, text_sketch_sde_feat_m, idxs)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # forward the positve image-text pair
        bs = image.size(0)
        
        output_pos = self.text_encoder(encoder_input_ids,
                                    attention_mask=text.attention_mask,
                                    encoder_hidden_states=image_embeds['last_hidden_state'],
                                    encoder_attention_mask=image_atts,
                                    return_dict=True,
                                    )
        if not self.negative_all_rank:
            # compute sample similarity
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t())

                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)

                sim_i2t = image_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ image_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            image_embeds_world = all_gather_with_grad(image_embeds)

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world['last_hidden_state'][neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from all ranks) for each image
            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])

        else:
            with torch.no_grad():
                mask = torch.eq(idx, idx.t())

                sim_i2t = image_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ image_feat.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

                    # select a negative image (from same rank) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds['last_hidden_state'][neg_idx])

            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from same rank) for each image    
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])
        
        
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds['last_hidden_state']], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        
        # with torch.no_grad():
        output_neg = self.text_encoder(text_ids_all,
                                    attention_mask=text_atts_all,
                                    encoder_hidden_states=image_embeds_all,
                                    encoder_attention_mask=image_atts_all,
                                    return_dict=True,
                                    )
    
        

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm*10
        
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        
        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr 

    @torch.no_grad()
    def _dequeue_and_enqueue_flip(self, image_feat, text_feat, sketch_sde_feat, text_sketch_sde_feat_m, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        sketch_sde_feats = concat_all_gather(sketch_sde_feat)
        text_sketch_sde_feat_m = concat_all_gather(text_sketch_sde_feat_m)

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.sketch_sde_queue[:, ptr:ptr + batch_size] = sketch_sde_feats.T
        self.text_sketch_sde_queue[:, ptr:ptr + batch_size] = text_sketch_sde_feat_m.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.ptr_queue[0] = ptr

def refine_model(pretrained='', **kwargs):
    model = fs2ksde_fflip_refine(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    if torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor.clone()

    return output  


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    # Queue the gathered tensors
    world_size = 1
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

def attention(self, embeds):
        layer_output = {}
        layer_output = embeds.intermediate_hidden_state 
        layer_output['layer_11'] = embeds.last_hidden_state[:,1:,:]
        layer_output['layer_3'] = layer_output['layer_3'][:,1:,:]
        layer_output['layer_5'] = layer_output['layer_5'][:,1:,:]
        layer_output['layer_7'] = layer_output['layer_7'][:,1:,:]
        
        layer_embeds_list = []
        layer_embeds_list.extend(output.permute(0,2,1) for output in layer_output.values())
        layer_embeds = self.attn(self.conv(torch.cat(layer_embeds_list, dim=1).reshape(16,768*4,14,14)))
        return layer_embeds