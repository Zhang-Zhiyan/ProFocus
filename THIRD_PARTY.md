# Third-party resources and attribution

The project's `LICENSE` records its Apache-2.0 license. It does not replace the terms of upstream code, pretrained weights, datasets, or other resources.

## Implementation provenance

The original archive credits the [SEVLM implementation](https://github.com/BetterZH/SEVLM-code) ([paper](https://arxiv.org/abs/2403.11150)) and [NLX-GPT](https://github.com/fawazsammani/nlxgpt). The decoder uses the Hugging Face GPT-2 interfaces, and the visual encoder uses [OpenAI CLIP](https://github.com/openai/CLIP). Preserve upstream copyright, license, and attribution notices when incorporating their implementations.

The archive does not provide a complete file-to-upstream mapping or the precise SEVLM/NLX-GPT source revisions. Maintainers should record that provenance and include any required upstream notices when confirming the final release. This document does not assert that every file inherited the project's license.

## NRC Valence, Arousal, and Dominance Lexicon

The NRC-VAD lexicon was created by Saif M. Mohammad at the National Research Council Canada. The v1 README in the supplied archive permits non-commercial research/education use and explicitly prohibits redistribution of the data. Consequently, lexicon data files are not included in this release. Obtain v1 directly from [the resource author's page](https://saifmohammad.com/WebPages/nrc-vad.html), retain its notices, and follow the accompanying terms.

The code expects the v1 bipolar-scale file. The current resource page may offer additional versions; changing versions or scales changes the model's input features.

Please cite:

```bibtex
@inproceedings{mohammad2018vad,
  title     = {Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words},
  author    = {Mohammad, Saif M.},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  year      = {2018}
}
```

## Images, annotations, and generated guidance

WikiArt image files and ArtEmis annotations are not bundled. Their acquisition and use remain subject to the respective resource terms. The original archive links to [WikiArt dataset information](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) and a [preprocessed ArtEmis dataset directory](https://huggingface.co/datasets/aphrorahf/artemis_for_ProFocus/tree/main/dataset); the latter was not verified as an immutable, complete release.

`llava1.6_results/` contains the generated guidance supplied with the source archive. Its metadata identifies the model as `llava-v1.6-mistral-7b-hf`. The generation script, exact prompts, full model revision, and data provenance manifest were not supplied. Retain the guidance's provenance and respect the underlying resource/model terms when redistributing or using it.

## Pretrained weights and caption evaluation

GPT-2 and CLIP pretrained weights are acquired separately. No ProFocus trained checkpoint is included. The repository does not grant additional rights to separately acquired weights.

Caption evaluation can use a supplied `cococaption` implementation or the Python 3 `pycocotools`/`pycocoevalcap` packages. Their licenses, Java-based components, model assets, and attribution requirements remain applicable. The exact evaluator used for the original experiments was not provided.

