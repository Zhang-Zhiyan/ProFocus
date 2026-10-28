import argparse
import os


def str_to_bool(value):
    """Parse explicit boolean CLI values such as --use_VAD false."""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ('true', '1', 'yes', 'y', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false.')


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fixed_encoder', type=str_to_bool, default=True, help='freeze the image encoder parameters if True')
    parser.add_argument('--load_from_epoch', type=str, default=None, help='checkpoint epoch to load (e.g. best-res), or None to train from scratch')
    parser.add_argument('--eval_steps', type=int, default=2000, help='evaluate on the val set every N training steps')
    parser.add_argument('--stop_after_evals', type=int, default=50, help='early stop after N evaluations without improvement')
    parser.add_argument('--len_prefix', type=int, default=0, help='length of the learned prefix')
    parser.add_argument('--prompt_text', type=str, default='the emotion is', help='prompt prefix for the emotion explanation')
    parser.add_argument('--max_seq_len', type=int, default=40, help='maximum length of the full explanation')
    parser.add_argument('--prefix_dim', type=int, default=768, help='dimension of the prefix embeddings')
    parser.add_argument('--use_VAD', type=str_to_bool, default=True, help='use VAD embeddings to enhance explanation embeddings')
    parser.add_argument('--use_cl_loss', type=str_to_bool, default=True, help='use the contrastive (CL) loss')
    parser.add_argument('--use_VAD_loss', type=str_to_bool, default=True, help='use the VAD loss on the explanation')
    parser.add_argument('--num_train_epochs', type=int, default=25, help='maximum number of training epochs')
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of the language-modeling (XE) loss')
    parser.add_argument('--cl_Loss_weight', type=float, default=1.5, help='weight of the contrastive (CL) loss')
    parser.add_argument('--VAD_Loss_weight', type=float, default=1.0, help='weight of the VAD loss')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate of the decoder')
    parser.add_argument('--en_LR', type=float, default=2e-5, help='learning rate of the image encoder')
    parser.add_argument('--emo_en_LR', type=float, default=4e-5, help='learning rate of the VAD encoder')
    parser.add_argument('--batch_size', type=int, default=8, help='per-GPU batch size')
    parser.add_argument('--VAD_En_Layers', type=int, default=3, help='number of layers in the VAD encoder')
    parser.add_argument('--VAD_En_Heads', type=int, default=1, help='number of attention heads in the VAD encoder')
    parser.add_argument('--use_text_guide', type=str_to_bool, default=True, help='use LLaVA three-stage text guidance')

    parser.add_argument('--base_model', type=str, default='gpt2', help='Hugging Face GPT-2 model ID or local model directory')
    parser.add_argument('--image_root', type=str, default=os.environ.get('PROFOCUS_IMAGE_ROOT', './wikiart_rescaled_max_size_to_600px_same_aspect_ratio'), help='root directory containing WikiArt images')

    parser.add_argument('--ckpt_path', type=str, default='./ckpts/', help='directory to save checkpoints')
    parser.add_argument('--caption_save_path', type=str, default='./results/', help='directory to save generated captions and scores')
    parser.add_argument('--annFileExp', type=str, default='./data/artEmisX_val_annot_exp.json', help='COCO-style ground-truth annotation file for evaluation (auto-generated from val data if missing)')
    parser.add_argument('--nle_data_train_path', type=str, default='./data/artEmisX_cl_train.json', help='training set JSON')
    parser.add_argument('--nle_data_val_path', type=str, default='./data/artEmisX_val.json', help='validation set JSON')
    parser.add_argument('--nle_data_test_path', type=str, default='./data/artEmisX_test.json', help='test set JSON')
    parser.add_argument('--llava_results_dir', type=str, default='./llava1.6_results', help='directory containing LLaVA-1.6 text guidance')

    args = parser.parse_args()

    return args
