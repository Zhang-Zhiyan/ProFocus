import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import os
import glob


from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
import json

from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from models.vad_encoder import Transformer
from utils import data_utils
from utils.eval_utils import top_filtering1
from utils.get_VADfeature import get_sentence_VAD
import sys
from utils.datasets import artEmisXTrainDataset, artEmisXEvalDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')  
except:
    pass 
from utils.opts import parse_opt
from co_attention_layer_improved import ThreeStageTextToImageFusion 

from utils.caption_metrics import COCO, COCOEvalCap


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    image_encoder_state_dict = None

    file_config = os.path.join(ckpt_path, model_name, 'config.json')
    with open(file_config, 'r') as f:
        config = json.load(f)

    opt.use_cl_loss = config['use_cl_loss']
    opt.use_VAD_loss = config['use_VAD_loss']

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
        use_output_projection=True,
    ).to(device)
    
    if 'fusion_module_state_dict' in ckpt_load:
        fusion_module_state_dict = ckpt_load['fusion_module_state_dict']
        fusion_module_loaded.load_state_dict(fusion_module_state_dict)
        print("========= load fusion_module ! ")
    else:
        print("========= fusion_module not found in checkpoint, using new initialization ! ")
    
    if 'emo_encoder_state_dict' in ckpt_load:
        opt.use_VAD = True
        emo_encoder_state_dict = ckpt_load['emo_encoder_state_dict']
        emo_encoder.load_state_dict(emo_encoder_state_dict)
        print("========= load emo_encoder ! ")
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model, 'emo_encoder': emo_encoder, 'fusion_module': fusion_module_loaded})
    else:
        model = nn.ModuleDict({'encoder': image_encoder, 'decoder': decoder_model, 'fusion_module': fusion_module_loaded})

    optimizer = get_optimizer(model, [opt.learning_rate, opt.en_LR, opt.emo_en_LR])
    try:
        optimizer.load_state_dict(ckpt_load['optimizer_state_dict'])
    except Exception as e:
        print(f"WARNING: optimizer_state_dict 加载失败（例如双层/单层切换后参数不一致），将使用新初始化的 optimizer。原因: {e}")
    start_epoch = int(ckpt_load['epoch']) + 1
    scheduler_dic = ckpt_load.get('scheduler', None)
    best_acc = ckpt_load['best_acc']
    best_res = ckpt_load['best_res']
    del ckpt_load
    torch.cuda.empty_cache()
    print("=========== load ckpt from epoch {}, best acc is: {} , \
          best res is: {}".format(start_epoch, best_acc, best_res))

    return tokenizer, model, optimizer, scheduler_dic, start_epoch, best_acc, best_res


def save_checkpoint(name, epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, best_acc, best_res, **kwargs):
    
    print(f"DEBUG: Saving checkpoint '{name}' to {ckpt_path}")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
        print(f"DEBUG: Created directory {ckpt_path}")

    model_name = 'nle_model_{}'.format(str(name))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(name) + '.tar'
    global save_tokenizer
    
    try:
        if not save_tokenizer:
            tokenizer.save_pretrained(os.path.join(ckpt_path, tokenizer_name))
            save_tokenizer = True
            print(f"DEBUG: Tokenizer saved to {os.path.join(ckpt_path, tokenizer_name)}")

        unwrapped_model.decoder.save_pretrained(os.path.join(ckpt_path, model_name), save_function=accelerator.save, safe_serialization=False)
        print(f"DEBUG: Decoder model saved to {os.path.join(ckpt_path, model_name)}")
        
        if opt.fixed_encoder:
            if opt.use_VAD:
                ckpt_load = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'emo_encoder_state_dict': unwrapped_model.emo_encoder.state_dict(),
                    'fusion_module_state_dict': unwrapped_model.fusion_module.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'best_res': best_res,
                    **kwargs
                }
            else:
                ckpt_load = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fusion_module_state_dict': unwrapped_model.fusion_module.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'best_res': best_res,
                    **kwargs
                }
        else:
            if opt.use_VAD:
                ckpt_load = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'encoder_state_dict': unwrapped_model.encoder.state_dict(),
                    'emo_encoder_state_dict': unwrapped_model.emo_encoder.state_dict(),
                    'fusion_module_state_dict': unwrapped_model.fusion_module.state_dict(),
                    'best_acc': best_acc,
                    'best_res': best_res,
                    **kwargs
                }
            else:
                ckpt_load = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'encoder_state_dict': unwrapped_model.encoder.state_dict(),
                    'fusion_module_state_dict': unwrapped_model.fusion_module.state_dict(),
                    'best_acc': best_acc,
                    'best_res': best_res,
                    **kwargs
                }

        accelerator.save(ckpt_load, os.path.join(ckpt_path, filename))
        print(f"DEBUG: Checkpoint stats saved to {os.path.join(ckpt_path, filename)}")
        
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()


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
        file_name = image_data["image_name"] + '.jpg'
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

def get_scores(annFile, resFile, save_scores_path, full_predictions):
    with open(opt.nle_data_val_path, 'r') as f:
        all_file = json.load(f)

    gt_answers = {}
    for key, value in all_file.items():
        has_max, maximizer = data_utils.proc_distribution(value['image_emotion_distribution'])
        if has_max:
            gt_answers[int(key)] = maximizer

    pred_answers = {}
    for item in full_predictions:
        if gt_answers.get(item['image_id']) is not None:
            pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()

    correct_keys = []
    for key, value in pred_answers.items():
        gt_answer = gt_answers[key]
        if value == gt_answer:
            correct_keys.append(key)

    if len(correct_keys) == 0:
        return -1, None

    acc = len(correct_keys) / len(gt_answers)

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    cocoEval.eval['sub_acc'] = acc
    with open(save_scores_path, 'w') as w:
        json.dump(cocoEval.eval, w)

    return acc, cocoEval.eval


def load_llava_answers(llava_results_dir, default_prompt_text="", train_data_path=None, val_data_path=None, test_data_path=None):
    answer_map = {}
    source_map = {}
    json_files = glob.glob(os.path.join(llava_results_dir, "*.json"))
    if not json_files:
        print(f"未在 {llava_results_dir} 下找到 LLaVA 结果 JSON，所有样本将使用默认文本作为引导。")
        return answer_map, source_map

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
    
    test_count = 0
    if test_data_path and os.path.exists(test_data_path):
        try:
            test_data = json.load(open(test_data_path, 'r'))
            for img_id, sample in test_data.items():
                img_name = sample.get('image_name', '')
                if img_name:
                    name_to_id_map[img_name] = str(img_id)
                    test_count += 1
            print(f"从测试集建立了 {test_count} 个 image_name -> image_id 的映射")
        except Exception as e:
            print(f"读取测试数据集建立映射时出错：{e}")
    
    if name_to_id_map:
        print(f"总共建立了 {len(name_to_id_map)} 个 image_name -> image_id 的映射")

    total_loaded = 0
    for json_file in json_files:
        try:
            json_filename = os.path.basename(json_file)
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
                    
                    stage_texts = [
                        visual_analysis if visual_analysis else default_prompt_text,
                        subject_description if subject_description else default_prompt_text,
                        striking_details if striking_details else default_prompt_text,
                    ]
                    answer_map[image_id] = stage_texts
                    source_map[image_id] = json_filename
                    total_loaded += 1

            elif isinstance(data, list):
                for item in data:
                    image_id = item.get("image_id")
                    results = item.get("results", [])
                    if image_id and results:
                        if isinstance(image_id, str) and '.' in image_id:
                            image_id = image_id.split('.')[0]
                        
                        stage_texts = []
                        for i in range(3):
                            ans = results[i].get("answer", "").strip() if i < len(results) else ""
                            stage_texts.append(ans if ans else default_prompt_text)
                        answer_map[str(image_id)] = stage_texts
                        source_map[str(image_id)] = json_filename
                        total_loaded += 1
                                
            elif isinstance(data, dict) and "image_id" in data:
                image_id = data.get("image_id")
                results = data.get("results", [])
                if image_id and results:
                    if isinstance(image_id, str) and '.' in image_id:
                        image_id = image_id.split('.')[0]
                    
                    stage_texts = []
                    for i in range(3):
                        ans = results[i].get("answer", "").strip() if i < len(results) else ""
                        stage_texts.append(ans if ans else default_prompt_text)
                    answer_map[str(image_id)] = stage_texts
                    source_map[str(image_id)] = json_filename
                    total_loaded += 1
        except Exception as e:
            print(f"读取 {json_file} 出错，已跳过：{e}")
            import traceback
            traceback.print_exc()

    print(f"共从 {len(json_files)} 个 JSON 中载入 {total_loaded} 条三阶段文本引导。")
    
    if test_data_path and os.path.exists(test_data_path):
        try:
            test_data = json.load(open(test_data_path, 'r'))
            test_ids = set(str(img_id) for img_id in test_data.keys())
            test_guide_count = sum(1 for img_id in answer_map.keys() if img_id in test_ids)
            print(f"其中测试集引导文本数量：{test_guide_count}/{len(test_ids)}")
        except Exception as e:
            print(f"统计测试集引导文本数量时出错：{e}")
    
    return answer_map, source_map


def generate_text_embeddings(three_stage_texts, tokenizer, decoder_model, device, max_length=64):
    batch_size = len(three_stage_texts)
    all_texts = (
        [t[0] for t in three_stage_texts] +
        [t[1] for t in three_stage_texts] +
        [t[2] for t in three_stage_texts]
    )
    all_embeddings, all_masks = _encode_texts(all_texts, tokenizer, decoder_model, device, max_length)

    return (
        (all_embeddings[:batch_size], all_embeddings[batch_size:2*batch_size], all_embeddings[2*batch_size:]),
        (all_masks[:batch_size], all_masks[batch_size:2*batch_size], all_masks[2*batch_size:])
    )


def _encode_texts(texts, tokenizer, decoder_model, device, max_length=64):
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

    attention_mask = encoded['attention_mask'].float()

    with torch.no_grad():
        text_embeddings = decoder_model.transformer.wte(encoded['input_ids'])

        text_embeddings = torch.nn.functional.layer_norm(
            text_embeddings, (text_embeddings.shape[-1],), eps=1e-5
        )

    return text_embeddings, attention_mask


def build_fused_encoder_states(img_embeddings, text_embeddings=None, text_attention_masks=None, fusion_module=None):
    if text_embeddings is None:
        return img_embeddings

    clip_prefix, clip_tokens = img_embeddings
    num_heads = fusion_module.stage1.fusion.num_attention_heads

    text_stage1, text_stage2, text_stage3 = text_embeddings
    text_stage1 = text_stage1.to(clip_tokens.device)
    text_stage2 = text_stage2.to(clip_tokens.device)
    text_stage3 = text_stage3.to(clip_tokens.device)

    mask_stage1, mask_stage2, mask_stage3 = text_attention_masks

    def to_attn_bias(mask):
        return (1.0 - mask[:, None, None, :].expand(-1, num_heads, 1, -1)) * (-1e9)

    enhanced_clip_tokens = fusion_module(
        clip_tokens,
        text_stage1, text_stage2, text_stage3,
        text_attention_mask1=to_attn_bias(mask_stage1),
        text_attention_mask2=to_attn_bias(mask_stage2),
        text_attention_mask3=to_attn_bias(mask_stage3),
    )

    return (clip_prefix, enhanced_clip_tokens)


def sample_sequences(model, tokenizer, loader):
    model.encoder.eval()
    model.decoder.eval()
    model.fusion_module.eval()
    if opt.use_VAD:
        model.emo_encoder.eval()

    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<prefix>', '<emotion>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 40
    from tqdm import tqdm
    
    total_samples = 0
    found_guide_count = 0

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids, VAD_features = batch
        bt = img.size(0)
        current_outputs = torch.full([bt, 1], -1).to(device)
        end_idx = [max_len] * bt
        always_exp = [False] * bt
        new_segment = torch.full([bt, 1], -1).to(device)

        if opt.use_text_guide:
            id_list = [str(idx.item()) for idx in img_id.view(-1)]
            total_samples += len(id_list)
            three_stage_texts = []
            for iid in id_list:
                stage_texts = answer_map.get(iid)
                if isinstance(stage_texts, list) and len(stage_texts) == 3:
                    three_stage_texts.append(stage_texts)
                    found_guide_count += 1
                else:
                    three_stage_texts.append([opt.prompt_text, opt.prompt_text, opt.prompt_text])
            text_embeddings, text_attention_masks = generate_text_embeddings(
                three_stage_texts, tokenizer, model.decoder, device)
        else:
            text_embeddings = None
            text_attention_masks = None

        with torch.no_grad():
            img_embeddings = model.encoder(img)
            encoder_hidden_states = build_fused_encoder_states(
                img_embeddings,
                text_embeddings if opt.use_text_guide else None,
                text_attention_masks=text_attention_masks if opt.use_text_guide else None,
                fusion_module=model.fusion_module if opt.use_text_guide else None,
            )

            for step in range(max_len + 1):
                if step == max_len:
                    break
                if opt.use_VAD:
                    (_, _, ef_output) = model.emo_encoder(VAD_features)
                else:
                    ef_output = None

                outputs = model.decoder(
                    input_ids=input_ids,
                    past_key_values=None,
                    attention_mask=(input_ids != tokenizer.pad_token_id).long(),
                    token_type_ids=segment_ids,
                    position_ids=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    labels=None,
                    use_cache=False,
                    return_dict=True,
                    emo_features=ef_output
                )

                lm_logits = outputs.logits[0]
                logits = lm_logits[:, -1, :] / temperature
                logits = top_filtering1(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
                updataed_VAD_features = []
                input_ids = torch.cat((input_ids, prev), dim=1)
                current_outputs = torch.cat((current_outputs, prev), dim=1)

                for j in range(bt):
                    if prev[j].item() in special_tokens_ids and end_idx[j] == max_len:
                        end_idx[j] = step + 1
                    decoded_sequences = [tokenizer.decode(wj, skip_special_tokens=False).lstrip() for wj in input_ids[j]]

                    VAD_values = get_sentence_VAD(decoded_sequences)
                    updataed_VAD_features.append(torch.tensor(VAD_values, dtype=torch.float))
                    if not always_exp[j]:
                        if prev[j].item() != because_token:
                            new_segment[j] = special_tokens_ids[-2]
                        else:
                            new_segment[j] = special_tokens_ids[-1]
                            always_exp[j] = True
                    else:
                        new_segment[j] = special_tokens_ids[-1]

                VAD_features = torch.stack(updataed_VAD_features).to(device).clone().detach()
                segment_ids = torch.cat((segment_ids, new_segment), dim=1)

            for k in range(bt):
                decoded_sequences = tokenizer.decode(current_outputs[k][1:end_idx[k]], skip_special_tokens=True).lstrip()
                results_full.append({"image_id": img_id[k].item(), "caption": decoded_sequences})
                if 'because' in decoded_sequences:
                    cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
                else:
                    cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])

                results_exp.append({"image_id": img_id[k].item(), "caption": cut_decoded_sequences})
    
    if opt.use_text_guide and total_samples > 0:
        guide_ratio = found_guide_count / total_samples * 100
        print(f"\n评估时引导文本使用情况：{found_guide_count}/{total_samples} ({guide_ratio:.1f}%) 样本成功加载引导文本")

    return results_full, results_exp


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    
    fusion_lr = learning_rate[0]

    if opt.fixed_encoder:
        if opt.use_VAD:
            optimizer = AdamW([
                {'params': [p for n, p in model.emo_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate[2]},
                {'params': [p for n, p in model.emo_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate[2]},
                {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate[0]},
                {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate[0]},
                {'params': [p for n, p in model.fusion_module.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': fusion_lr},
                {'params': [p for n, p in model.fusion_module.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': fusion_lr}])
        else:
            optimizer_grouped_parameters_decoder = [
                {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in model.fusion_module.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': fusion_lr},
                {'params': [p for n, p in model.fusion_module.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': fusion_lr}]
            optimizer = AdamW(optimizer_grouped_parameters_decoder, lr=learning_rate[0])

    else:
        if opt.use_VAD:
            optimizer = AdamW([
                {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate[1]},
                {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate[1]},
                {'params': [p for n, p in model.emo_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate[2]},
                {'params': [p for n, p in model.emo_encoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate[2]},
                {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate[0]},
                {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate[0]},
                {'params': [p for n, p in model.fusion_module.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': fusion_lr},
                {'params': [p for n, p in model.fusion_module.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': fusion_lr}])
        else:
            optimizer = AdamW([
                {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate[1]},
                {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate[1]},
                {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': learning_rate[0]},
                {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': learning_rate[0]},
                {'params': [p for n, p in model.fusion_module.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': fusion_lr},
                {'params': [p for n, p in model.fusion_module.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': fusion_lr}])

    return optimizer


def add_summary_value(writer, keys, value, iteration):
    with writer.as_default():
        tf.summary.scalar(keys, value, step=iteration)
        writer.flush()


accelerator = Accelerator(mixed_precision="no")  
device = accelerator.device
opt = parse_opt()
print(opt)

no_sample = True
top_k = 0
top_p = 0.9
temperature = 1

if (not os.path.exists(opt.ckpt_path)):
    os.makedirs(opt.ckpt_path, exist_ok=True)
if (not os.path.exists(opt.caption_save_path)):
    os.makedirs(opt.caption_save_path, exist_ok=True)

img_size = 224
weight_decay = 0
gradient_accumulation_steps = 1
start_epoch = 0
save_tokenizer = False
best_acc = 0.0
best_res = 0.0
best_acc_epoch = 0
best_res_epoch = 0
no_improvment = 0
EF_DIM = 3

image_encoder = ImageEncoder(device).to(device)
encoder_out_dim = 768
if opt.fixed_encoder:
    change_requires_grad(image_encoder, False)
else:
    change_requires_grad(image_encoder, True)

fusion_module = ThreeStageTextToImageFusion(
    hidden_size=768,
    num_attention_heads=12,
    dropout_prob=0.1,
    intermediate_size=3072,
    use_output_projection=True,
).to(device)

if opt.use_VAD:
    emo_encoder = Transformer(EF_DIM, num_layers=opt.VAD_En_Layers, nhead=opt.VAD_En_Heads, dim_feedforward=128)
    emo_encoder = emo_encoder.to(device)

if opt.load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch, best_acc, best_res = load_checkpoint(
        ckpt_path=opt.ckpt_path, epoch=opt.load_from_epoch)
else:
    tokenizer = GPT2Tokenizer.from_pretrained(opt.base_model)
    orig_num_tokens = len(tokenizer.encoder)
    num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                   'additional_special_tokens': ['<prefix>', '<emotion>',
                                                                                 '<explanation>']})
    assert len(tokenizer) == orig_num_tokens + 4
    config = AutoConfig.from_pretrained(opt.base_model)
    setattr(config, 'img_size', None)
    setattr(config, 'max_seq_len', None)
    setattr(config, 'len_prefix', None)
    setattr(config, 'prefix_size', None)
    setattr(config, 'encoder_dim', None)
    setattr(config, 'use_cl_loss', None)
    setattr(config, 'use_VAD_loss', None)
    config.img_size = img_size
    config.max_seq_len = opt.max_seq_len
    config.len_prefix = opt.len_prefix
    config.add_cross_attention = True
    config.prefix_size = opt.prefix_dim
    config.encoder_dim = encoder_out_dim
    config.use_cl_loss = opt.use_cl_loss
    config.use_VAD_loss = opt.use_VAD_loss

    decoder_model = GPT2LMHeadModel.from_pretrained(opt.base_model, config=config)

    decoder_model.resize_token_embeddings(len(tokenizer))
    decoder_model = decoder_model.to(device)

    if opt.use_VAD:
        model = nn.ModuleDict({
            'encoder': image_encoder, 
            'decoder': decoder_model, 
            'emo_encoder': emo_encoder,
            'fusion_module': fusion_module
        })
    else:
        model = nn.ModuleDict({
            'encoder': image_encoder, 
            'decoder': decoder_model,
            'fusion_module': fusion_module
        })
    optimizer = get_optimizer(model, [opt.learning_rate, opt.en_LR, opt.emo_en_LR])

print("Model Setup Ready...")

answer_map, _ = load_llava_answers(
    opt.llava_results_dir,
    default_prompt_text=opt.prompt_text,
    train_data_path=opt.nle_data_train_path,
    val_data_path=opt.nle_data_val_path,
    test_data_path=opt.nle_data_test_path
)

img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                         std=[0.26862954, 0.26130258, 0.27577711])])

train_dataset = artEmisXTrainDataset(path=opt.nle_data_train_path,
                                     transform=img_transform,
                                     tokenizer=tokenizer,
                                     max_seq_len=opt.max_seq_len,
                                     len_prefix=opt.len_prefix,
                                     prompt_text=opt.prompt_text,
                                     image_root=opt.image_root,
)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=6)

val_dataset = artEmisXEvalDataset(path=opt.nle_data_val_path,
                                  transform=img_transform,
                                  tokenizer=tokenizer,
                                  max_seq_len=opt.max_seq_len,
                                  len_prefix=opt.len_prefix,
                                  prompt_text=opt.prompt_text,
                                  image_root=opt.image_root)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=64,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=6)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
print("训练集/测试集大小: {}/{} ".format(len(train_dataset), len(val_dataset)))

num_batch = len(train_loader)
t_total = (num_batch // gradient_accumulation_steps) * opt.num_train_epochs
warmup_steps = 0

def get_linear_decay_scheduler(optimizer, num_training_steps):
    def lr_lambda(current_step):
        if current_step >= num_training_steps:
            return 0.0
        return 1.0 - (current_step / num_training_steps)

    return LambdaLR(optimizer, lr_lambda)

scheduler = get_linear_decay_scheduler(optimizer, t_total)

if opt.load_from_epoch is not None and scheduler_dic is not None:
    scheduler.load_state_dict(scheduler_dic)

tf_summary_writer = tf.summary.create_file_writer(opt.ckpt_path)

for epoch in range(start_epoch, opt.num_train_epochs):
    if opt.fixed_encoder:
        model.encoder.eval()
    else:
        model.encoder.train()
    model.decoder.train()
    model.fusion_module.train()
    if opt.use_VAD:
        model.emo_encoder.train()

    accum_loss = 0
    accum_loss_xe_text = 0
    accum_loss_cl = 0
    accum_loss_vad = 0

    for step, batch in enumerate(train_loader):
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, did, input_ids, labels, segment_ids, VAD_labels = batch

        if opt.use_text_guide:
            id_list = [str(idx.item()) for idx in did.view(-1)]
            three_stage_texts = []
            for iid in id_list:
                stage_texts = answer_map.get(iid)
                if isinstance(stage_texts, list) and len(stage_texts) == 3:
                    three_stage_texts.append(stage_texts)
                else:
                    three_stage_texts.append([opt.prompt_text, opt.prompt_text, opt.prompt_text])

            text_embeddings_bt, text_attention_masks_bt = generate_text_embeddings(
                three_stage_texts, tokenizer, model.decoder, device)
        else:
            text_embeddings_bt = None
            text_attention_masks_bt = None

        img_shape = img.shape
        lab_shape = labels.shape
        VAD_labels_shape = VAD_labels.shape

        img = img.view(-1, img_shape[-3], img_shape[-2], img_shape[-1])
        input_ids = input_ids.view(-1, lab_shape[-1])
        labels = labels.view(-1, lab_shape[-1])
        segment_ids = segment_ids.view(-1, lab_shape[-1])
        VAD_labels = VAD_labels.view(-1, VAD_labels_shape[-2], VAD_labels_shape[-1])

        if opt.use_text_guide and text_embeddings_bt is not None:
            text_embeddings_bt2 = tuple(
                emb.repeat_interleave(2, dim=0) for emb in text_embeddings_bt
            )
            text_attention_masks_bt2 = tuple(
                mask.repeat_interleave(2, dim=0) for mask in text_attention_masks_bt
            )
            del text_embeddings_bt, text_attention_masks_bt
        else:
            text_embeddings_bt2 = None
            text_attention_masks_bt2 = None

        if opt.fixed_encoder:
            with torch.no_grad():
                img_embeddings = model.encoder(img)
        else:
            img_embeddings = model.encoder(img)

        encoder_hidden_states = build_fused_encoder_states(
            img_embeddings,
            text_embeddings_bt2 if opt.use_text_guide else None,
            text_attention_masks=text_attention_masks_bt2 if opt.use_text_guide else None,
            fusion_module=model.fusion_module if opt.use_text_guide else None,
        )

        if opt.use_VAD:
            (_, _, ef_output) = model.emo_encoder(VAD_labels)
        else:
            ef_output = None

        outputs = model.decoder(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
            token_type_ids=segment_ids,
            position_ids=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            labels=labels,
            use_cache=False,
            return_dict=True,
            alpha=opt.alpha,
            cl_Loss_weight=opt.cl_Loss_weight,
            emo_features=ef_output,
            VAD_labels=VAD_labels,
            VAD_Loss_weight=opt.VAD_Loss_weight,
        )

        loss = outputs.loss[0]
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告: 检测到 NaN/Inf loss，跳过本步更新。Step: {step}, Epoch: {epoch}")
            optimizer.zero_grad()
            continue

        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        
        loss_value = loss.item()
        accum_loss += loss_value

        loss_xe_text = outputs.loss[1][0]
        if loss_xe_text is not None:
            accum_loss_xe_text += (loss_xe_text / gradient_accumulation_steps).item()

        loss_cl = outputs.loss[2]
        if loss_cl is not None:
            accum_loss_cl += (loss_cl / gradient_accumulation_steps).item()

        loss_vad = outputs.loss[3]
        if loss_vad is not None:
            accum_loss_vad += (loss_vad / gradient_accumulation_steps).item()

        if step % 1000 == 0:
            fusion_grad_norm = 0
            fusion_param_norm = 0
            for name, param in model.fusion_module.named_parameters():
                if param.grad is not None:
                    fusion_grad_norm += param.grad.norm().item() ** 2
                fusion_param_norm += param.norm().item() ** 2
            fusion_grad_norm = fusion_grad_norm ** 0.5
            fusion_param_norm = fusion_param_norm ** 0.5
            
            print(f"Epoch {epoch}/{opt.num_train_epochs}, Step {step}/{len(train_loader)}, "
                  f"Accumulated Loss:{accum_loss:.4f}, "
                  f"Accumulated XE Loss:{accum_loss_xe_text:.4f}, "
                  f"Accumulated CL Loss:{accum_loss_cl:.4f}, "
                  f"Accumulated VAD Loss:{accum_loss_vad:.4f}, "
                  f"Fusion Grad Norm:{fusion_grad_norm:.4f}, "
                  f"Fusion Param Norm:{fusion_param_norm:.4f}")

        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            max_grad_norm = 1
            if hasattr(accelerator, 'clip_grad_norm_'):
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            if grad_norm is not None:
                grad_norm_value = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
                if (isinstance(grad_norm_value, float) and 
                    (not (grad_norm_value == grad_norm_value) or
                     grad_norm_value == float('inf') or 
                     grad_norm_value == float('-inf') or 
                     grad_norm_value > 100)):
                    print(f"警告: 梯度异常 (norm={grad_norm_value:.2f})，跳过参数更新。Step: {step}, Epoch: {epoch}")
                    optimizer.zero_grad()
                    continue
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            add_summary_value(tf_summary_writer, 'sum_loss', accum_loss, int(epoch * num_batch + step))
            add_summary_value(tf_summary_writer, 'loss_xe_text', accum_loss_xe_text, int(epoch * num_batch + step))
            add_summary_value(tf_summary_writer, 'loss_cl', accum_loss_cl, int(epoch * num_batch + step))
            add_summary_value(tf_summary_writer, 'loss_vad', accum_loss_vad, int(epoch * num_batch + step))
            add_summary_value(tf_summary_writer, 'lr', optimizer.state_dict()['param_groups'][0]['lr'],
                              int(epoch * num_batch + step))
            accum_loss = 0
            accum_loss_xe_text = 0
            accum_loss_cl = 0
            accum_loss_vad = 0

        if int(epoch * num_batch + (step + 1)) % opt.eval_steps == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                results_full, results_exp = sample_sequences(unwrapped_model, tokenizer, val_loader)
                unf_resFileExp = os.path.join(
                    opt.caption_save_path, f'unf_captions_exp_{epoch}_{step}_step.json'
                )
                unf_resFileFull = os.path.join(
                    opt.caption_save_path, f'unf_captions_full_{epoch}_{step}_step.json'
                )
                save_scores_pathExp = os.path.join(
                    opt.caption_save_path, f'scores_exp_{epoch}_{step}_step.json'
                )
                with open(unf_resFileExp, 'w') as w:
                    json.dump(results_exp, w)
                with open(unf_resFileFull, 'w') as w:
                    json.dump(results_full, w)

                if not os.path.exists(opt.annFileExp):
                    try:
                        os.makedirs(os.path.dirname(opt.annFileExp), exist_ok=True)
                    except:
                        print('目录已存在!')
                    get_annFile(opt.nle_data_val_path, opt.annFileExp)
                    print(f"已自动生成验证集标注文件: {opt.annFileExp}")

                acc, res = get_scores(opt.annFileExp, unf_resFileExp, save_scores_pathExp, results_full)
                print("emotion acc is: {}".format(acc))

                if opt.fixed_encoder:
                    model.encoder.eval()
                else:
                    model.encoder.train()
                model.decoder.train()
                model.fusion_module.train()
                if opt.use_VAD:
                    model.emo_encoder.train()

                if res is None:
                    continue
                if acc > best_acc or res['Bleu_4'] > best_res:
                    if acc > best_acc:
                        best_acc = acc
                        best_acc_epoch = epoch
                        no_improvment = 0
                        save_checkpoint('best-acc', epoch, unwrapped_model, optimizer, tokenizer, scheduler,
                                        opt.ckpt_path, best_acc, best_res)
                        print(" best acc model is saved !")
                    if res['Bleu_4'] > best_res:
                        best_res = res['Bleu_4']
                        best_res_epoch = epoch
                        no_improvment = 0
                        save_checkpoint('best-res', epoch, unwrapped_model, optimizer, tokenizer, scheduler,
                                        opt.ckpt_path, best_acc, best_res)
                        print("best result model is saved !")
                else:
                    no_improvment += 1
                    print("{} times no improvment !".format(no_improvment))
                if no_improvment > opt.stop_after_evals:
                    print("sys.exit !")
                    print("best acc is {} at epoch {}, best B4 is {} at epoch {}.".format(
                        best_acc, best_acc_epoch, best_res, best_res_epoch))
                    sys.exit()