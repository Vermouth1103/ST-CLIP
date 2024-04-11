import os.path as osp

import torch
torch.cuda.current_device()
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from trainers.trainer_build import TRAINER_REGISTRY
from trainers.data_manager import DataManager

from trainers.trainer import TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.attention import MultiHeadAttention
import copy

from dassl.utils import read_image
import json
from dassl.data.transforms.transforms import _build_transform_test
from torchvision.transforms import Normalize

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):

    def __init__(self, clip_model, device):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.device = device

    def forward(self, prompts, tokenized_prompts):
        bs, n_cls, n_len, n_dim = prompts.shape
        x = prompts.reshape(-1, n_len, n_dim) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)      # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)      # LND -> NLD
        x = self.ln_final(x).type(self.dtype)     

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1).repeat(bs)] @ self.text_projection
        x = x.reshape(bs, n_cls, -1)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([[pos / np.power(10000, 2 * i / d_model) for i in range(d_model)] if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()               # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())

class Encoder_block(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.attn = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads).half()
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//16),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model//16, self.d_model)
        ).half()
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x = self.attn(x)
        x = self.layer_norm(self.ffn(x) + x)
        return x

def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]).half()

class PromptLearner(nn.Module):

    def __init__(self, cfg, classname_list, clip_model, device, block_num=6):
        super().__init__()
        self.n_masks = len(classname_list)
        n_cls_list = [len(classname) for classname in classname_list]
        n_ctx = cfg.TRAINER.STCLIP.N_CTX
        ctx_init = cfg.TRAINER.STCLIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.device = device
        self.block_num = block_num
        self.cfg = cfg
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # TODO
            # init context vectors using trajectory embedding
            pass
        else:
            # random initialization
            if cfg.TRAINER.STCLIP.CSC:
                # TODO
                print("Initializing class-specific contexts")
            else:
                print("Initializing a generic context")
                ctx_vector = torch.empty(n_ctx*self.n_masks, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vector, std=0.02)
                self.ctx_vector = nn.Parameter(ctx_vector)  # (n_ctx*self.n_masks, ctx_dim)

            prompt_prefix = " ".join(["X"] * n_ctx) + " %s."

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        prompts_list = list()
        name_lens_list = list()
        for cn_index, classname in enumerate(classname_list):
            classname_list[cn_index] = [name.replace("_", " ") for name in classname_list[cn_index]]
            name_lens_list.append([len(_tokenizer.encode(name)) for name in classname_list[cn_index]])
            prompts_list.append([prompt_prefix % name for name in classname_list[cn_index]])

        tokenized_prompts_list = list()
        for pl_index, prompts in enumerate(prompts_list):
            temp = torch.cat([clip.tokenize(prompt) for prompt in prompts])
            tokenized_prompts_list.append(temp)
    
        embedding_list = list()
        with torch.no_grad():
            for tpl_index, tokenized_prompts in enumerate(tokenized_prompts_list):
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
                embedding_list.append(embedding)


        self.token_prefix_list = list()  # SOS
        self.token_suffix_list = list()  # CLS, EOS
        for index in range(len(embedding_list)):
            self.token_prefix_list.append(embedding_list[index][:, :1, :])  # (n_cls, 1, ctx_dim)
            self.token_suffix_list.append(embedding_list[index][:, 1+n_ctx:, :])  # (n_cls, rest, ctx_dim)

        self.n_cls_list = n_cls_list
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts_list = tokenized_prompts_list  # list(torch.Tensor)
        self.name_lens_list = name_lens_list
        self.class_token_position = cfg.TRAINER.STCLIP.CLASS_TOKEN_POSITION

        self.link_property_num = cfg.DATASET.LINK_PROPERTY_NUM

        self.link_id_emb_layer = nn.Embedding(cfg.DATASET.LINK_ID_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()
        self.fc_emb_layer = nn.Embedding(cfg.DATASET.FC_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()
        self.lane_emb_layer = nn.Embedding(cfg.DATASET.LANE_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()
        self.speed_class_emb_layer = nn.Embedding(cfg.DATASET.SPEED_CLASS_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()
        self.length_emb_layer = nn.Embedding(cfg.DATASET.LENGTH_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()
        self.out_degree_emb_layer = nn.Embedding(cfg.DATASET.OUT_DEGREE_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()

        self.route_cnt_emb_layer = nn.Embedding(cfg.DATASET.ROUTE_CNT_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()
        self.medium_speed_emb_layer = nn.Embedding(cfg.DATASET.MEDIUM_SPEED_NUM, cfg.DATASET.LINK_PROPERTY_DIM).half()

        self.ffn = nn.Sequential(
            nn.Linear(self.link_property_num * cfg.DATASET.LINK_PROPERTY_DIM, ctx_dim//16),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim//16, ctx_dim)
        ).half()

        self.para_attn_layer = MultiHeadAttention(d_model=ctx_dim, num_heads=8).half()

        self.pos_encoder = PositionalEncoding(d_model=ctx_dim).half()

        self.attn_layer = MultiHeadAttention(d_model=ctx_dim, num_heads=8).half()
        self.attn_block = clones(self.attn_layer, self.block_num)

        self.ffn_after = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim//16),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim//16, ctx_dim)
        ).half()

    def forward(self, traj_property):

        bs = traj_property.shape[0]

        traj_feature = list()
        for index in range(traj_property.shape[1]):
            link_id_emb = self.link_id_emb_layer(traj_property[:, index, 0])
            fc_emb = self.fc_emb_layer(traj_property[:, index, 1])
            lane_emb = self.lane_emb_layer(traj_property[:, index, 2])
            speed_class_emb = self.speed_class_emb_layer(traj_property[:, index, 3])
            length_emb = self.length_emb_layer(traj_property[:, index, 4])
            out_degree_emb = self.out_degree_emb_layer(traj_property[:, index, 5])

            route_cnt_emb = self.route_cnt_emb_layer(traj_property[:, index, 9])
            medium_speed_emb = self.medium_speed_emb_layer(traj_property[:, index, 11])

            link_feature = torch.cat([link_id_emb, fc_emb, lane_emb, speed_class_emb, \
                length_emb, out_degree_emb, route_cnt_emb, medium_speed_emb], axis=1)
            
            traj_feature.append(link_feature)

        traj_feature = torch.stack(traj_feature)  # (seq_len, bs, emb_num*emb_dim)
        traj_feature = traj_feature.permute(1, 0, 2)    # (bs, seq_len, emb_num*emb_dim)
        traj_feature = self.ffn(traj_feature)   # (bs, seq_len, ctx_dim)

        traj_feature = self.pos_encoder(traj_feature)
        
        for layer in self.attn_block:
            traj_feature, _ = layer(traj_feature, traj_feature, traj_feature)  # (bs, seq_len, ctx_dim)

        traj_feature = traj_feature[:, 1, :].repeat(1, self.n_ctx).reshape(-1, self.n_ctx, self.ctx_dim).unsqueeze(1)    # (bs, 1, n_ctx, ctx_dim)

        ctx_vector = self.ctx_vector.unsqueeze(0)
        ctx_vector, attn = self.para_attn_layer(ctx_vector, ctx_vector, ctx_vector)
        attn_path = osp.join(self.cfg.OUTPUT_DIR, "attention.npy")
        with open(f"{attn_path}", "wb") as f:
            np.save(f, attn.detach().cpu().numpy())
        ctx_vector = ctx_vector.squeeze()

        ctx_list = list()
        for index in range(self.n_masks):
            if ctx_vector.dim() == 2:
                ctx_list.append(ctx_vector[self.n_ctx*index:self.n_ctx*(index+1), :].unsqueeze(0).expand(self.n_cls_list[index], -1, -1).clone())  # (n_cls_list[index], n_ctx, ctx_dim)

        prompts_list = list()
        
        for index in range(len(ctx_list)):
            prompts = torch.cat(
                [
                    self.token_prefix_list[index].unsqueeze(0).repeat(bs, 1, 1, 1).to(self.device),  # (bs, n_cls_list[index], 1, dim)
                    ctx_list[index].unsqueeze(0).repeat(bs, 1, 1, 1) + traj_feature,                # (bs, n_cls_list[index], n_ctx, dim)
                    self.token_suffix_list[index].unsqueeze(0).repeat(bs, 1, 1, 1).to(self.device)   # (bs, n_cls_list[index], *, dim)
                ],
                dim=2
            )
            prompts_list.append(prompts)
    
        return prompts_list

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classname_list, clip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classname_list, clip_model, device)
        self.tokenized_prompts_list = self.prompt_learner.tokenized_prompts_list
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.cfg = cfg

        if "EVAL_ONLY" in self.cfg.MODEL:
            self.text_embedding_list = list()
            self.pred_list = list()

    def forward(self, image, traj_property):
        logit_scale = self.logit_scale.exp()
        logits_list = list()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.unsqueeze(1)

        prompts_list = self.prompt_learner(traj_property)
        tokenized_prompts_list = self.tokenized_prompts_list

        for index in range(len(tokenized_prompts_list)):
            text_features = self.text_encoder(prompts_list[index], tokenized_prompts_list[index])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if "EVAL_ONLY" in self.cfg.MODEL:
                if index == len(self.text_embedding_list):
                    self.text_embedding_list.append(list())
                    self.pred_list.append(list())
                self.text_embedding_list[index].append(text_features.data.cpu().numpy())

            logits = logit_scale * torch.bmm(image_features, text_features.permute(0, 2, 1)).squeeze()
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            logits_list.append(logits)
            if "EVAL_ONLY" in self.cfg.MODEL:
                self.pred_list[index].extend(logits.max(axis=1)[1].data.cpu().numpy().tolist())
        
        return logits_list

@TRAINER_REGISTRY.register()
class STCLIP(TrainerX):

    def build_data_loader(self):

        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_class_list = dm.num_class_list
        self.lab2cname_list = dm.lab2cname_list  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classname_list = self.dm.dataset.classname_list

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")

        clip_model = load_clip_to_cpu(cfg)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classname_list, clip_model, self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Number of Parameters: ", total_params)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label_list, traj_property = self.parse_batch_train(batch)
        logits_list = self.model(image, traj_property)

        # TODO 
        # loss function with label_list and logits_list
        loss = 0
        for index in range(label_list.shape[1]):
            loss += F.cross_entropy(logits_list[index], label_list[:, index])
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item()
        }
        for index in range(label_list.shape[1]):
            loss_summary[f"acc_{index}"] = compute_accuracy(logits_list[index], label_list[:, index])[0].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label_list = batch["label_list"].to(self.device)
        traj_property = batch["traj_property"].to(self.device)

        return image, label_list, traj_property

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label_list, traj_property = self.parse_batch_test(batch)
            logits_list = self.model_inference(image, traj_property)
            self.evaluator.process(logits_list, label_list, self.device)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_inference(self, image_path):
        """inference"""
        self.set_model_mode("eval")

        link_profile_dict_path = osp.join(self.cfg.DATASET.ROOT, self.cfg.DATASET.NAME.lower(), "link_profile_dict.json")
        with open(link_profile_dict_path, "r") as f:
            link_property_dict = json.load(f)
        link_traj_dict_path = osp.join(self.cfg.DATASET.ROOT, self.cfg.DATASET.NAME.lower(), "link_traj_dict.json")
        with open(link_traj_dict_path, "r") as f:
            link_traj_dict = json.load(f)

        def _transform_image(tfm, img0):
            img_list = []
            k_tfm = 1

            for k in range(k_tfm):
                img_list.append(tfm(img0))

            img = img_list
            if len(img) == 1:
                img = img[0]

            return img

        complete_prompt = "The current road is in the %s scene. The surface is %s, and the width is %s. It's %s to pass through."
        scene_list = ["field", "cars", "alley", "stall", "unknown"]
        surface_list = ["normal", "broken", "soil", "unknown"]
        width_list = ["normal", "narrow", "extremely_narrow", "unknown"]
        through_list = ["easy", "hard", "extremely_hard"]
        kind_list = [scene_list, surface_list, width_list, through_list]

        with open(image_path, "r") as f:
            image_list = f.readlines()

        # build transform
        choices = self.cfg.INPUT.TRANSFORMS
        target_size = f"{self.cfg.INPUT.SIZE[0]}x{self.cfg.INPUT.SIZE[1]}"
        normalize = Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        transform = _build_transform_test(self.cfg, choices, target_size, normalize)

        f = open(osp.join(self.cfg.OUTPUT_DIR, "output.txt"), "w")

        for index in range(len(image_list)):
            image_path = image_list[index].strip()
            f.write(f"Image path: {image_path}\n")
            f.flush()

            img0 = read_image(image_path)

            output = {}
            if isinstance(transform, (list, tuple)):
                for i, tfm in enumerate(transform):
                    img = _transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = _transform_image(transform, img0)
                output["img"] = img
            image = output["img"].unsqueeze(0).to(self.device)

            link_traj = link_traj_dict[image_path.split("/")[-1]]
            link_property = np.array(link_property_dict[str(link_traj["link_id"])])
            prev_link_property = np.array(link_property_dict[str(link_traj["prev_link_id"])])
            next_link_property = np.array(link_property_dict[str(link_traj["next_link_id"])])
            traj_property = np.array([prev_link_property, link_property, next_link_property])
            traj_property = torch.IntTensor(traj_property).unsqueeze(0).to(self.device)

            try:
                logits_list = self.model(image, traj_property)
            except:
                continue

            answer = []
            for index in range(len(logits_list)):
                logits = logits_list[index].data.cpu().numpy()
                answer.append(kind_list[index][np.argmax(logits)])

            print(f"Trajectory: {str(link_traj['prev_link_id'])}, {str(link_traj['link_id'])}, {str(link_traj['next_link_id'])}")
            print(f"Description: {complete_prompt % tuple(answer)}")

            f.write(f"Description: {complete_prompt % tuple(answer)}\n")
            f.flush()
        
        f.close()

    def model_inference(self, image, traj_property):
        return self.model(image, traj_property)

    def parse_batch_test(self, batch):
        image = batch["img"].to(self.device)
        label_list = batch["label_list"].to(self.device)
        traj_property = batch["traj_property"].to(self.device)

        return image, label_list, traj_property