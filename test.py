import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import os
from transformers import GPT2Tokenizer, AutoConfig
import json
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from models.vad_encoder import Transformer
from utils import data_utils
from utils.data_utils import *
from utils.eval_utils import top_filtering1
from utils.get_VADfeature import get_sentence_VAD
import time
import sys
import numpy as np
from utils.datasets import artEmisXEvalDataset
import argparse
from utils.opts import str_to_bool
import glob
from utils.caption_metrics import COCO, COCOEvalCap
from co_attention_layer_improved import ThreeStageTextToImageFusion


ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--len_prefix", type=int, default=0,help="the length of prefix")
    parser.add_argument("--prompt_text",type=str,default=" the emotion is")
    parser.add_argument("--max_seq_len",type=int,default=40,help="the length of full explanation")
    parser.add_argument("--use_VAD", type=str_to_bool, default=True, help="use VAD embeddings to enhance explanation embeddings")
    parser.add_argument("--use_text_guide", type=str_to_bool, default=True, help="use LLaVA three-stage text guidance")
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--VAD_En_Layers",type=int,default=3)
    parser.add_argument("--VAD_En_Heads",type=int,default=1)

    parser.add_argument('--load_from_epoch',type=str, default='best-acc', help="checkpoint epoch to load (e.g. best-res)")
    parser.add_argument('--image_root', type=str, default=os.environ.get('PROFOCUS_IMAGE_ROOT', './wikiart_rescaled_max_size_to_600px_same_aspect_ratio'), help='root directory containing WikiArt images')
    parser.add_argument("--ckpt_path",type=str,default='./ckpts/')
    parser.add_argument("--caption_save_path",type=str,default="./test-results/")
    parser.add_argument("--annFileExp",type=str,default='./data/artEmisX_test_annot_exp.json')
    parser.add_argument("--nle_data_test_path",type=str,default='./data/artEmisX_test.json')
    parser.add_argument('--llava_results_dir', type=str, default='./llava1.6_results',
                       help='Directory containing LLaVA results for text guidance')

    args = parser.parse_args()
    
    return args

def get_annFile(nle_data_test_path, annFileExp):
    with open(nle_data_test_path, 'r') as f:
        data = json.load(f)

    output_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": []
    }

    annotation_id = 0

    for image_id, image_data in data.items():
        file_name = image_data["image_name"]+'jpg'
        output_data["images"].append({
            "id": int(image_id),
            "file_name": file_name
        })

        for explanation in image_data["explanations"]:
            output_data["annotations"].append({
                "image_id": int(image_id),
                "id": annotation_id,
                "caption": explanation
            })
            annotation_id += 1

    with open(annFileExp, 'w') as f:
        json.dump(output_data, f, indent=4)

def load_checkpoint(image_encoder, emo_encoder, ckpt_path, epoch, device):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    image_encoder_state_dict =None

    
    file_config = os.path.join(ckpt_path, model_name, 'config.json')
    config = json.load(open(file_config,'r'))


    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(ckpt_path, tokenizer_name))
    decoder_model = GPT2LMHeadModel.from_pretrained(os.path.join(ckpt_path, model_name)).to(device)


    
    ckpt_load = torch.load(os.path.join(ckpt_path, filename))

    if 'encoder_state_dict' in ckpt_load:
        image_encoder_state_dict = ckpt_load['encoder_state_dict']
        image_encoder.load_state_dict(image_encoder_state_dict)
        print("========= load encoder ! ")

    fusion_module_loaded = ThreeStageTextToImageFusion(
        hidden_size=768,
        num_attention_heads=12,
        dropout_prob=0.1,
        intermediate_size=3072,
        use_output_projection=True
    ).to(device)
    
    if 'fusion_module_state_dict' in ckpt_load:
        fusion_module_state_dict = ckpt_load['fusion_module_state_dict']
        fusion_module_loaded.load_state_dict(fusion_module_state_dict)
        print("========= load fusion_module ! ")
    else:
        print("========= fusion_module not found in checkpoint, using new initialization ! ")

    if 'emo_encoder_state_dict' in ckpt_load:
        emo_encoder_state_dict = ckpt_load['emo_encoder_state_dict']
        emo_encoder.load_state_dict(emo_encoder_state_dict)
        print("========= load emo_encoder ! ")
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model, 'emo_encoder':emo_encoder, 'fusion_module': fusion_module_loaded})
    else:
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model, 'fusion_module': fusion_module_loaded})

    best_acc = ckpt_load['best_acc']
    best_res=ckpt_load['best_res']
    torch.cuda.empty_cache()
    print("=========== load ckpt from epoch {}, best acc is: {} , \
          best res is: {}".format(epoch, best_acc, best_res))

    return tokenizer, model, best_acc,best_res

def load_llava_answers(llava_results_dir, use_three_stage=True, default_prompt_text="", train_data_path=None, val_data_path=None, test_data_path=None):
    answer_map = {}
    guide_text_sources = {}
    json_files = glob.glob(os.path.join(llava_results_dir, "*.json"))
    if not json_files:
        print(f"未在 {llava_results_dir} 下找到 LLaVA 结果 JSON，所有样本将使用默认文本作为引导。")
        return answer_map, guide_text_sources

    name_to_id_map = {}
    if train_data_path and os.path.exists(train_data_path):
        try:
            train_data = json.load(open(train_data_path, 'r'))
            for img_id, sample in train_data.items():
                img_name = sample.get('image_name', '')
                if img_name:
                    name_to_id_map[img_name] = str(img_id)
        except Exception as e:
            print(f"读取训练数据集建立映射时出错：{e}")
    
    if val_data_path and os.path.exists(val_data_path):
        try:
            val_data = json.load(open(val_data_path, 'r'))
            for img_id, sample in val_data.items():
                img_name = sample.get('image_name', '')
                if img_name:
                    name_to_id_map[img_name] = str(img_id)
        except Exception as e:
            print(f"读取验证数据集建立映射时出错：{e}")
    
    if test_data_path and os.path.exists(test_data_path):
        try:
            test_data = json.load(open(test_data_path, 'r'))
            for img_id, sample in test_data.items():
                img_name = sample.get('image_name', '')
                if img_name:
                    name_to_id_map[img_name] = str(img_id)
        except Exception as e:
            print(f"读取测试数据集建立映射时出错：{e}")
    
    if name_to_id_map:
        print(f"建立了 {len(name_to_id_map)} 个 image_name -> image_id 的映射")

    total_loaded = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and "analyses" in data:
                analyses = data.get("analyses", [])
                for item in analyses:
                    image_name = item.get("image_name", "")
                    status = item.get("status", "")
                    
                    if not image_name or status != "success":
                        continue
                    
                    if image_name.endswith('.jpg'):
                        image_name_no_ext = image_name[:-4]
                    else:
                        image_name_no_ext = image_name
                    
                    image_id = name_to_id_map.get(image_name_no_ext)
                    if not image_id:
                        continue
                    
                    visual_analysis = item.get("visual_analysis", "").strip()
                    subject_description = item.get("subject_description", "").strip()
                    striking_details = item.get("striking_details", "").strip()
                    
                    if use_three_stage:
                        stage_texts = []
                        stage_texts.append(visual_analysis if visual_analysis else default_prompt_text)
                        stage_texts.append(subject_description if subject_description else default_prompt_text)
                        stage_texts.append(striking_details if striking_details else default_prompt_text)
                        
                        if len(stage_texts) == 3:
                            answer_map[image_id] = stage_texts
                            guide_text_sources[image_id] = json_file
                            total_loaded += 1
                    else:
                        if visual_analysis:
                            answer_map[image_id] = visual_analysis
                            guide_text_sources[image_id] = json_file
                            total_loaded += 1

            elif isinstance(data, list):
                for item in data:
                    image_id = item.get("image_id")
                    results = item.get("results", [])
                    if image_id and results:
                        if isinstance(image_id, str) and '.' in image_id:
                            image_id = image_id.split('.')[0]
                        
                        if use_three_stage:
                            stage_texts = []
                            for i in range(3):
                                if i < len(results):
                                    ans = results[i].get("answer", "").strip()
                                    if ans:
                                        stage_texts.append(ans)
                                    else:
                                        stage_texts.append(default_prompt_text)
                                else:
                                    stage_texts.append(default_prompt_text)
                            
                            if len(stage_texts) == 3:
                                answer_map[str(image_id)] = stage_texts
                                guide_text_sources[str(image_id)] = json_file
                                total_loaded += 1
                        else:
                            ans = results[0].get("answer", "").strip()
                            if ans:
                                answer_map[str(image_id)] = ans
                                guide_text_sources[str(image_id)] = json_file
                                total_loaded += 1
        except Exception as e:
            print(f"读取 {json_file} 出错，已跳过：{e}")

    if use_three_stage:
        print(f"共从 {len(json_files)} 个 JSON 中载入 {total_loaded} 条三阶段文本引导。")
    else:
        print(f"共从 {len(json_files)} 个 JSON 中载入 {total_loaded} 条文本引导。")
    return answer_map, guide_text_sources

def _generate_single_stage_embeddings(texts, tokenizer, decoder_model, device, max_length=64):
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        raise ValueError("texts cannot be empty")
    
    texts = [str(t) if t is not None else "" for t in texts]
    
    encoded = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding='max_length'
    ).to(device)

    attention_mask = encoded['attention_mask']

    with torch.no_grad():
        text_embeddings = decoder_model.transformer.wte(encoded['input_ids'])
        text_embeddings = torch.nn.functional.layer_norm(
            text_embeddings, (text_embeddings.shape[-1],), eps=1e-5
        )

    return text_embeddings, attention_mask

def generate_text_embeddings(texts, tokenizer, decoder_model, device, max_length=64):
    if not texts or len(texts) == 0:
        raise ValueError("texts cannot be empty")
    
    if isinstance(texts[0], list) and len(texts[0]) == 3:
        batch_size = len(texts)
        stage1_texts = [t[0] for t in texts]
        stage2_texts = [t[1] for t in texts]
        stage3_texts = [t[2] for t in texts]
        
        all_texts = stage1_texts + stage2_texts + stage3_texts
        all_embeddings, all_masks = _generate_single_stage_embeddings(all_texts, tokenizer, decoder_model, device, max_length)
        
        embeddings_stage1 = all_embeddings[:batch_size]
        embeddings_stage2 = all_embeddings[batch_size:2*batch_size]
        embeddings_stage3 = all_embeddings[2*batch_size:]
        masks_stage1 = all_masks[:batch_size]
        masks_stage2 = all_masks[batch_size:2*batch_size]
        masks_stage3 = all_masks[2*batch_size:]
        
        return (embeddings_stage1, embeddings_stage2, embeddings_stage3), (masks_stage1, masks_stage2, masks_stage3)
    else:
        embeddings, masks = _generate_single_stage_embeddings(texts, tokenizer, decoder_model, device, max_length)
        return embeddings, masks

def build_fused_encoder_states(img_embeddings, text_embeddings=None, text_attention_masks=None, fusion_module=None, use_three_stage=True):
    if text_embeddings is None:
        return img_embeddings

    clip_prefix, clip_tokens = img_embeddings

    if fusion_module is not None:
        if hasattr(fusion_module, 'stage1') and hasattr(fusion_module, 'stage2') and hasattr(fusion_module, 'stage3'):
            if use_three_stage:
                text_stage1, text_stage2, text_stage3 = text_embeddings
                text_stage1 = text_stage1.to(clip_tokens.device)
                text_stage2 = text_stage2.to(clip_tokens.device)
                text_stage3 = text_stage3.to(clip_tokens.device)

                batch_size, num_regions, _ = clip_tokens.size()

                if text_attention_masks is not None:
                    mask_stage1, mask_stage2, mask_stage3 = text_attention_masks
                    mask_stage1 = mask_stage1.to(clip_tokens.device)
                    mask_stage2 = mask_stage2.to(clip_tokens.device)
                    mask_stage3 = mask_stage3.to(clip_tokens.device)
                    
                    mask_stage1_expanded = mask_stage1[:, None, None, :]
                    mask_stage2_expanded = mask_stage2[:, None, None, :]
                    mask_stage3_expanded = mask_stage3[:, None, None, :]
                    is_padding1 = (mask_stage1_expanded == 0)
                    is_padding2 = (mask_stage2_expanded == 0)
                    is_padding3 = (mask_stage3_expanded == 0)
                    text_attention_mask1 = is_padding1.float() * (-1e9)
                    text_attention_mask2 = is_padding2.float() * (-1e9)
                    text_attention_mask3 = is_padding3.float() * (-1e9)
                else:
                    _, num_words1, _ = text_stage1.size()
                    _, num_words2, _ = text_stage2.size()
                    _, num_words3, _ = text_stage3.size()
                    text_attention_mask1 = torch.zeros((batch_size, 1, 1, num_words1), device=clip_tokens.device, dtype=clip_tokens.dtype)
                    text_attention_mask2 = torch.zeros((batch_size, 1, 1, num_words2), device=clip_tokens.device, dtype=clip_tokens.dtype)
                    text_attention_mask3 = torch.zeros((batch_size, 1, 1, num_words3), device=clip_tokens.device, dtype=clip_tokens.dtype)

                enhanced_clip_tokens = fusion_module(
                    clip_tokens,
                    text_stage1,
                    text_stage2,
                    text_stage3,
                    text_attention_mask1=text_attention_mask1,
                    text_attention_mask2=text_attention_mask2,
                    text_attention_mask3=text_attention_mask3
                )
                fused_tokens = enhanced_clip_tokens
            else:
                if isinstance(text_embeddings, tuple):
                    text_embeddings = text_embeddings[0].to(clip_tokens.device)
                else:
                    text_embeddings = text_embeddings.to(clip_tokens.device)
                
                batch_size, num_regions, _ = clip_tokens.size()
                
                if text_attention_masks is not None:
                    if isinstance(text_attention_masks, tuple):
                        mask = text_attention_masks[0].to(clip_tokens.device)
                    else:
                        mask = text_attention_masks.to(clip_tokens.device)
                    mask_expanded = mask[:, None, None, :]
                    is_padding = (mask_expanded == 0)
                    text_attention_mask = is_padding.float() * (-1e9)
                else:
                    _, num_words, _ = text_embeddings.size()
                    text_attention_mask = torch.zeros((batch_size, 1, 1, num_words), device=clip_tokens.device, dtype=clip_tokens.dtype)
    
                enhanced_clip_tokens = fusion_module.stage1(
                    clip_tokens,
                    text_embeddings,
                    text_attention_mask
                )
                fused_tokens = enhanced_clip_tokens
        else:
            text_embeddings = text_embeddings.to(clip_tokens.device) if not isinstance(text_embeddings, tuple) else text_embeddings[0].to(clip_tokens.device)
            batch_size, num_regions, _ = clip_tokens.size()
            
            if text_attention_masks is not None:
                mask = text_attention_masks if not isinstance(text_attention_masks, tuple) else text_attention_masks[0]
                mask = mask.to(clip_tokens.device)
                mask_expanded = mask[:, None, None, :]
                is_padding = (mask_expanded == 0)
                text_attention_mask = is_padding.float() * (-1e9)
            else:
                _, num_words, _ = text_embeddings.size()
                text_attention_mask = torch.zeros((batch_size, 1, 1, num_words), device=clip_tokens.device, dtype=clip_tokens.dtype)

            enhanced_clip_tokens = fusion_module(
                clip_tokens,
                text_embeddings,
                text_attention_mask=text_attention_mask
            )
            fused_tokens = enhanced_clip_tokens
    else:
        if isinstance(text_embeddings, tuple):
            text_stage1, text_stage2, text_stage3 = text_embeddings
            all_text_tokens = torch.cat([
                text_stage1.to(clip_tokens.device),
                text_stage2.to(clip_tokens.device),
                text_stage3.to(clip_tokens.device)
            ], dim=1)
        else:
            all_text_tokens = text_embeddings.to(clip_tokens.device)
        fused_tokens = torch.cat([clip_tokens, all_text_tokens], dim=1)
    
    return (clip_prefix, fused_tokens)


def synchronize_if_cuda(device):
    if not torch.cuda.is_available():
        return
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if str(device_type).startswith('cuda'):
        torch.cuda.synchronize(device)

def get_scores(annFile, resFile, save_scores_path,  full_predictions, nle_data_test_path):
    all_file = json.load(open(nle_data_test_path, 'r'))
    
    gt_answers = {}
    for key,value in all_file.items():
        has_max,maximizer=data_utils.proc_distribution(value['origin_emotion_distribution'])
        if has_max:
            gt_answers[int(key)]= maximizer


    pred_answers = {}
    for item in full_predictions:
        if gt_answers.get(item['image_id']) is not None:
            pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()
        
    correct_keys = []
    for key,value in pred_answers.items():
        gt_answer = gt_answers[key]
        if value == gt_answer:
            correct_keys.append(key)
   
    if len(correct_keys) ==0 :
        return -1,None 
      
    acc = len(correct_keys)/len(gt_answers) 

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    cocoEval.eval['sub_acc']=acc
    with open(save_scores_path, 'w') as w:
        json.dump(cocoEval.eval, w)
    
    return acc, cocoEval.eval
    
def sample_sequences(model, tokenizer, loader, device, opt, answer_map=None):
    no_sample = True
    top_k =  0
    top_p =  0.9
    temperature = 1
    
    model.encoder.eval()
    model.decoder.eval()
    if opt.use_text_guide:
        model.fusion_module.eval()
    if opt.use_VAD:
        model.emo_encoder.eval()
    
    all_prefix_vis = []
    all_encoder_hs = []
    all_prefix_exp = []
    all_exp_embeded = []
    all_emotion_logits_9 = []
    all_emotion_probs_9 = []
        
    results_exp = []
    results_full = []
    total_inference_time = 0.0
    total_inference_samples = 0
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<prefix>', '<emotion>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    emotion_token_ids = []
    for emo in ARTEMIS_EMOTIONS:
        token_text = " something" if emo == "something else" else f" {emo}"
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) == 0:
            ids = tokenizer.encode(emo, add_special_tokens=False)
        emotion_token_ids.append(ids[0])
    max_len = 40
    len_prefix = opt.len_prefix
    
    from tqdm import tqdm
    for i,batch in  tqdm (enumerate(loader),total=len(loader)):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids, VAD_features = batch
        bt = img.size(0)
        synchronize_if_cuda(device)
        batch_infer_start = time.perf_counter()
        current_outputs = torch.full([bt,1],-1).to(device)
        end_idx=[max_len]*bt
        always_exp=[False]*bt
        new_segment=torch.full([bt,1],-1).to(device)
        batch_emotion_logits_9 = None
        batch_emotion_probs_9 = None
        
        text_embeddings = None
        text_attention_masks = None
        use_three_stage = False
        if opt.use_text_guide and answer_map is not None:
            id_list = [str(idx.item()) for idx in img_id.view(-1)]
            
            use_three_stage = isinstance(answer_map.get(id_list[0], opt.prompt_text), list)
            
            if use_three_stage:
                three_stage_texts = []
                for iid in id_list:
                    stage_texts = answer_map.get(iid, [opt.prompt_text, opt.prompt_text, opt.prompt_text])
                    if not isinstance(stage_texts, list) or len(stage_texts) != 3:
                        stage_texts = [opt.prompt_text, opt.prompt_text, opt.prompt_text]
                    three_stage_texts.append(stage_texts)
                
                if len(three_stage_texts) > 0:
                    text_embeddings, text_attention_masks = generate_text_embeddings(
                        three_stage_texts, tokenizer, model.decoder, device, max_length=64
                    )
                else:
                    default_texts = [[opt.prompt_text, opt.prompt_text, opt.prompt_text]]
                    text_embeddings, text_attention_masks = generate_text_embeddings(
                        default_texts, tokenizer, model.decoder, device, max_length=64
                    )
            else:
                single_texts = [answer_map.get(iid, opt.prompt_text) for iid in id_list]
                text_embeddings, text_attention_masks = generate_text_embeddings(
                    single_texts, tokenizer, model.decoder, device, max_length=64
                )
        
        with torch.no_grad():
            img_embeddings = model.encoder(img)
            
            encoder_hidden_states = build_fused_encoder_states(
                img_embeddings, 
                text_embeddings if opt.use_text_guide else None,
                text_attention_masks=text_attention_masks if opt.use_text_guide else None,
                fusion_module=model.fusion_module if opt.use_text_guide else None,
                use_three_stage=use_three_stage if opt.use_text_guide else False
            )
            
            model.decoder.prefix_vis_list = []
            model.decoder.encoder_hs_list = []
            model.decoder.prefix_exp_list = []
            model.decoder.exp_embeded_list = []
            
            for step in range(max_len + 1):
                if step == max_len:
                    break
                if opt.use_VAD:
                    (_, _, ef_output) = model.emo_encoder(VAD_features)
                else:
                    ef_output = None

                outputs = model.decoder(input_ids=input_ids, 
                            past_key_values=None, 
                            attention_mask=None, 
                            token_type_ids=segment_ids, 
                            position_ids=None, 
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=None, 
                            labels=None, 
                            use_cache=False, 
                            return_dict=True,
                            emo_features=ef_output,
                            collect_vectors=False)
                
                lm_logits = outputs.logits[0]
                logits = lm_logits[:, -1, :] / temperature
                if step == 0:
                    emo_logits = logits[:, emotion_token_ids]
                    emo_probs = torch.softmax(emo_logits, dim=-1)
                    batch_emotion_logits_9 = emo_logits.detach().cpu().numpy().astype(np.float32)
                    batch_emotion_probs_9 = emo_probs.detach().cpu().numpy().astype(np.float32)
                logits = top_filtering1(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1) 
                updataed_VAD_features = []
                input_ids = torch.cat((input_ids, prev), dim = 1)
                current_outputs  = torch.cat((current_outputs, prev), dim = 1)
                
                for j in range(bt):
                    if prev[j].item() in special_tokens_ids and end_idx[j]==max_len:
                        end_idx[j]=step+1
                    decoded_sequences = [tokenizer.decode(wj, skip_special_tokens=False).lstrip() for wj in input_ids[j]]
        
                    VAD_values= get_sentence_VAD(decoded_sequences)
                    updataed_VAD_features.append(torch.tensor(VAD_values,dtype=torch.float))
                    if not always_exp[j]:
                        if prev[j].item() != because_token:
                            new_segment[j] = special_tokens_ids[-2]   
                        else:
                            new_segment[j] = special_tokens_ids[-1]   
                            always_exp[j] = True
                    else:
                        new_segment[j] = special_tokens_ids[-1]   
               
                VAD_features = torch.stack(updataed_VAD_features).to(device).clone().detach()       
                segment_ids = torch.cat((segment_ids,new_segment),dim=1)
            
            for k in range(bt):
                sample_input_ids = input_ids[k:k+1, :end_idx[k]]
                sample_segment_ids = segment_ids[k:k+1, :end_idx[k]]
                
                if isinstance(encoder_hidden_states, tuple):
                    sample_encoder_hs = (encoder_hidden_states[0][k:k+1], encoder_hidden_states[1][k:k+1])
                else:
                    sample_encoder_hs = encoder_hidden_states[k:k+1]
                
                if opt.use_VAD:
                    sample_decoded = tokenizer.decode(input_ids[k][1:end_idx[k]], skip_special_tokens=False).lstrip()
                    sample_decoded_tokens = sample_decoded.split()
                    sample_VAD_values = get_sentence_VAD(sample_decoded_tokens)
                    sample_VAD_tensor = torch.tensor(sample_VAD_values, dtype=torch.float).unsqueeze(0).to(device)
                    (_, _, ef_output_sample) = model.emo_encoder(sample_VAD_tensor)
                    sample_ef_output = ef_output_sample[:, :end_idx[k]]
                else:
                    sample_ef_output = None
                
                model.decoder.prefix_vis_list = []
                model.decoder.encoder_hs_list = []
                model.decoder.prefix_exp_list = []
                model.decoder.exp_embeded_list = []
                
                _ = model.decoder(
                    input_ids=sample_input_ids,
                    past_key_values=None,
                    attention_mask=None,
                    token_type_ids=sample_segment_ids,
                    position_ids=None,
                    encoder_hidden_states=sample_encoder_hs,
                    encoder_attention_mask=None,
                    labels=None,
                    use_cache=False,
                    return_dict=True,
                    emo_features=sample_ef_output,
                    collect_vectors=True
                )
                
                if len(model.decoder.prefix_vis_list) > 0:
                    all_prefix_vis.append(model.decoder.prefix_vis_list[0])
                    all_encoder_hs.append(model.decoder.encoder_hs_list[0])
                    all_prefix_exp.append(model.decoder.prefix_exp_list[0])
                    all_exp_embeded.append(model.decoder.exp_embeded_list[0])
                    if batch_emotion_logits_9 is not None and batch_emotion_probs_9 is not None:
                        all_emotion_logits_9.append(batch_emotion_logits_9[k])
                        all_emotion_probs_9.append(batch_emotion_probs_9[k])
                    else:
                        all_emotion_logits_9.append(np.zeros((9,), dtype=np.float32))
                        all_emotion_probs_9.append(np.zeros((9,), dtype=np.float32))
                else:
                    all_prefix_vis.append(np.zeros(768, dtype=np.float32))
                    all_encoder_hs.append(np.zeros(768, dtype=np.float32))
                    all_prefix_exp.append(np.zeros(768, dtype=np.float32))
                    all_exp_embeded.append(np.zeros(768, dtype=np.float32))
                    all_emotion_logits_9.append(np.zeros((9,), dtype=np.float32))
                    all_emotion_probs_9.append(np.zeros((9,), dtype=np.float32))
                
            for k in range(bt):
                decoded_sequences = tokenizer.decode(current_outputs[k][1:end_idx[k]], skip_special_tokens=True).lstrip()
                results_full.append({"image_id": img_id[k].item(), "caption": decoded_sequences})
                if 'because' in decoded_sequences:
                    cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
                else:
                    cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
                
                results_exp.append({"image_id": img_id[k].item(), "caption": cut_decoded_sequences})

        synchronize_if_cuda(device)
        batch_infer_time = time.perf_counter() - batch_infer_start
        total_inference_time += batch_infer_time
        total_inference_samples += bt
        print(
            f"[Inference Time] batch {i + 1}/{len(loader)} | "
            f"batch_size={bt} | "
            f"batch_time={batch_infer_time:.4f}s | "
            f"avg_per_image={batch_infer_time / max(bt, 1):.4f}s"
        )

    inference_time_stats = {
        'total_inference_time_sec': total_inference_time,
        'total_inference_samples': total_inference_samples,
        'avg_inference_time_per_image_sec': (
            total_inference_time / total_inference_samples if total_inference_samples > 0 else 0.0
        ),
        'throughput_images_per_sec': (
            total_inference_samples / total_inference_time if total_inference_time > 0 else 0.0
        ),
    }
              
    return (
        results_full, results_exp,
        all_prefix_vis, all_encoder_hs, all_prefix_exp, all_exp_embeded,
        all_emotion_logits_9, all_emotion_probs_9,
        inference_time_stats,
    )

def test_one_model():

    accelerator = Accelerator()
    device = accelerator.device
    opt = parse_opt()

    if (not os.path.exists(opt.ckpt_path)):
        os.makedirs(opt.ckpt_path, exist_ok=True)
    if (not os.path.exists(opt.caption_save_path)):
        os.makedirs(opt.caption_save_path, exist_ok=True)

    img_size = 224
    EF_DIM = 3


    image_encoder = ImageEncoder(device).to(device)

    emo_encoder = None
    if opt.use_VAD:
        emo_encoder = Transformer(EF_DIM, num_layers=opt.VAD_En_Layers, nhead=opt.VAD_En_Heads, dim_feedforward = 128)
        emo_encoder = emo_encoder.to(device)

    if opt.load_from_epoch is not None:
        tokenizer, model, best_acc, best_res = load_checkpoint(image_encoder, emo_encoder, ckpt_path=opt.ckpt_path, epoch=opt.load_from_epoch, device=device)

    print("Model Setup Ready...")

    if opt.use_text_guide:
        answer_map, guide_text_sources = load_llava_answers(
            opt.llava_results_dir, 
            use_three_stage=True, 
            default_prompt_text=opt.prompt_text,
            test_data_path=opt.nle_data_test_path
        )
        print(f"文本引导功能已启用，共加载 {len(answer_map)} 条引导文本。")
    else:
        answer_map = {}
        guide_text_sources = {}
        print("文本引导功能已禁用，将使用默认 prompt_text。")

    img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = artEmisXEvalDataset(path = opt.nle_data_test_path,      
                                transform = img_transform, 
                                tokenizer = tokenizer, 
                                max_seq_len = opt.max_seq_len,
                                len_prefix=opt.len_prefix,
                                prompt_text=opt.prompt_text,
                                image_root=opt.image_root)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size = opt.batch_size*2, 
                                            shuffle=False, 
                                            pin_memory=True,
                                            num_workers=6)


    nle_data_test = json.load(open(opt.nle_data_test_path, 'r'))
    emotion_map = {}
    for key, value in nle_data_test.items():
        has_max, maximizer = data_utils.proc_distribution(value['origin_emotion_distribution'])
        if has_max:
            emotion_map[int(key)] = maximizer
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model) 
    if accelerator.is_main_process:
        (
            results_full, results_exp,
            prefix_vis_list, encoder_hs_list, prefix_exp_list, exp_embeded_list,
            emotion_logits_9_list, emotion_probs_9_list,
            inference_time_stats,
        ) = sample_sequences(
            unwrapped_model, tokenizer, test_loader, device, opt, answer_map=answer_map
        )
        print(f"\n推理耗时统计:")
        print(f"  - 总推理时间: {inference_time_stats['total_inference_time_sec']:.4f} 秒")
        print(f"  - 推理图片数: {inference_time_stats['total_inference_samples']}")
        print(f"  - 单张图片平均推理时间: {inference_time_stats['avg_inference_time_per_image_sec']:.4f} 秒")
        print(f"  - 平均吞吐量: {inference_time_stats['throughput_images_per_sec']:.4f} 张/秒")
        
        print(f"\n特征向量统计:")
        print(f"  - prefix_vis: {len(prefix_vis_list)} 个样本")
        print(f"  - encoder_hs: {len(encoder_hs_list)} 个样本")
        print(f"  - prefix_exp: {len(prefix_exp_list)} 个样本")
        print(f"  - exp_embeded: {len(exp_embeded_list)} 个样本")
        print(f"  - emotion_logits_9: {len(emotion_logits_9_list)} 个样本")
        print(f"  - emotion_probs_9: {len(emotion_probs_9_list)} 个样本")
        print(f"  - results_full: {len(results_full)} 个样本")
        print(f"  - results_exp: {len(results_exp)} 个样本")
        
        min_len = min(len(prefix_vis_list), len(results_full))
        if len(prefix_vis_list) != len(results_full):
            print(f"\n警告: 特征向量数量 ({len(prefix_vis_list)}) 与结果数量 ({len(results_full)}) 不匹配！")
            print(f"将只保存前 {min_len} 个匹配的样本...")
        
        results_full_with_features = []
        results_exp_with_features = []
        
        for idx in range(min_len):
            image_id = results_full[idx]['image_id']
            emotion_label = emotion_map.get(image_id, 'something else')
            
            result_full = results_full[idx].copy()
            result_full['features'] = {
                'prefix_vis': prefix_vis_list[idx].tolist() if isinstance(prefix_vis_list[idx], np.ndarray) else prefix_vis_list[idx],
                'encoder_hs': encoder_hs_list[idx].tolist() if isinstance(encoder_hs_list[idx], np.ndarray) else encoder_hs_list[idx],
                'prefix_exp': prefix_exp_list[idx].tolist() if isinstance(prefix_exp_list[idx], np.ndarray) else prefix_exp_list[idx],
                'exp_embeded': exp_embeded_list[idx].tolist() if isinstance(exp_embeded_list[idx], np.ndarray) else exp_embeded_list[idx],
                'emotion_logits_9': emotion_logits_9_list[idx].tolist() if isinstance(emotion_logits_9_list[idx], np.ndarray) else emotion_logits_9_list[idx],
                'emotion_probs_9': emotion_probs_9_list[idx].tolist() if isinstance(emotion_probs_9_list[idx], np.ndarray) else emotion_probs_9_list[idx],
                'emotion_label': emotion_label
            }
            results_full_with_features.append(result_full)
            
            result_exp = results_exp[idx].copy()
            result_exp['features'] = {
                'prefix_vis': prefix_vis_list[idx].tolist() if isinstance(prefix_vis_list[idx], np.ndarray) else prefix_vis_list[idx],
                'encoder_hs': encoder_hs_list[idx].tolist() if isinstance(encoder_hs_list[idx], np.ndarray) else encoder_hs_list[idx],
                'prefix_exp': prefix_exp_list[idx].tolist() if isinstance(prefix_exp_list[idx], np.ndarray) else prefix_exp_list[idx],
                'exp_embeded': exp_embeded_list[idx].tolist() if isinstance(exp_embeded_list[idx], np.ndarray) else exp_embeded_list[idx],
                'emotion_logits_9': emotion_logits_9_list[idx].tolist() if isinstance(emotion_logits_9_list[idx], np.ndarray) else emotion_logits_9_list[idx],
                'emotion_probs_9': emotion_probs_9_list[idx].tolist() if isinstance(emotion_probs_9_list[idx], np.ndarray) else emotion_probs_9_list[idx],
                'emotion_label': emotion_label
            }
            results_exp_with_features.append(result_exp)
        
        unf_resFileExp = os.path.join(opt.caption_save_path, 'unf_captions_exp_' + opt.load_from_epoch + '.json')
        unf_resFileFull = os.path.join(opt.caption_save_path, 'unf_captions_full_' + opt.load_from_epoch + '.json')
        
        with open(unf_resFileExp, 'w') as w:
            json.dump(results_exp_with_features, w, indent=2)     
        with open(unf_resFileFull, 'w') as w:
            json.dump(results_full_with_features, w, indent=2)
        
        print(f"\n特征向量已保存到JSON文件:")
        print(f"  - 完整结果（带特征向量）: {unf_resFileFull}")
        print(f"  - 解释结果（带特征向量）: {unf_resFileExp}")
        

        save_scores_pathExp = os.path.join(opt.caption_save_path, 'scores_exp_' + opt.load_from_epoch + '.json')
        if (not os.path.exists(opt.annFileExp)):
            try: 
                os.makedirs(os.path.dirname(opt.annFileExp))
            except:
                print('dir exist!')
            get_annFile(opt.nle_data_test_path, opt.annFileExp)
        acc, res = get_scores(opt.annFileExp, unf_resFileExp, save_scores_pathExp, results_full, opt.nle_data_test_path)            
        print("emotion acc is: {}".format(acc))
        

if __name__ == "__main__":
    test_one_model()