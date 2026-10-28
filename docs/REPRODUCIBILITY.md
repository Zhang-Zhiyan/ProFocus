# Reproducibility notes

This is a source release assembled from the supplied `SEVLMcodemain` archive. File names are retained to make the original implementation recognizable. Portability changes make resource paths configurable and remove assumptions about the original server. They do not establish reproduction of the paper's experiments.

## What is available

- Training and checkpoint evaluation entry points, the model modules, and the progressive fusion module.
- Supplied LLaVA-1.6 three-stage guidance JSON files and method illustrations.
- Dependency specifications, resource preparation instructions, and the project's existing license.

Static checks can establish syntax and resource structure. They cannot establish numerical correctness, training convergence, evaluation equivalence, or agreement with a paper result. This release has no validated end-to-end experiment or claimed reproduced score.

## Resources required for a complete run

1. The exact preprocessed ArtEmis annotations and version-specific train/validation/test manifests, including preprocessing code or a documented conversion procedure and checksums.
2. Corresponding WikiArt images with confirmed image preprocessing.
3. The bipolar-scale NRC-VAD v1 file, downloaded by each user under its terms, and the NLTK language resources.
4. The GPT-2 and CLIP pretrained weights, with their precise revisions/cache contents recorded.
5. A working caption evaluator with Java and its required assets. For paper comparisons, record the evaluator fork, commit, tokenizer, metric versions, and any metric-specific downloads.
6. Complete trained checkpoints for each published dataset/experiment, including tokenizer, decoder configuration and weights, fusion and VAD states, and a fine-tuned image encoder state when applicable.

The supplied archive does not contain the annotations, image files, trained checkpoints, original environment export, HAC generation script/prompts, or a confirmed original evaluator fork. The annotation link in the original README is retained as a lead, not a verified immutable dataset release.

## Preserved differences requiring author confirmation

The following behavior is present in the supplied source and is not silently standardized in this release:

- **Image normalization:** training uses CLIP mean/std values; testing uses ImageNet mean/std values. The intended evaluation preprocessing must be confirmed against the paper experiment and checkpoint.
- **Prompt:** the training default is `the emotion is`, while the evaluation default has a leading space: ` the emotion is`. Record the exact prompt and use explicit `--prompt_text` values after confirming the intended setup.
- **Missing fusion state:** checkpoint loading can initialize new fusion parameters if `fusion_module_state_dict` is absent. Verify that released checkpoints contain the trained fusion state before reporting results.
- **Missing guidance:** a missing file or unmatched image can lead to fallback prompt guidance. Check coverage for every split; a completed run alone does not confirm that HAC guidance was used.
- **Contrastive pairs:** training samples a different emotion from the same image's annotations. Each training record needs at least two distinct emotion labels. The archive supplies no conversion script documenting this filtering.
- **Emotion scoring:** validation reads `image_emotion_distribution`, whereas testing reads `origin_emotion_distribution`; scoring uses the greater-than-0.5 majority policy in `data_utils.proc_distribution`. Both score functions return `(-1, None)` when there are no correct predictions. These conventions require confirmation before interpreting results.
- **Evaluator choice:** the compatibility fallback to Python 3 packages has not been shown to match the original evaluator numerically.

No paper performance figures are asserted here. These issues need experimental confirmation, rather than assumptions based on the presence of source code.

## Defaults versus an experiment record

The CLI defaults in `utils/opts.py` and model constructors describe the current source configuration. They are not a substitute for saved commands and configuration from the paper runs. The original complete Python environment, CUDA/driver details, hardware, number of devices, random seeds, dataset hashes, model revisions, and run logs were not supplied. The training and test entry points do not set a global experiment seed.

To document a validated experiment, preserve:

- The repository commit, complete launch command, environment export, device details, and random seed policy.
- Dataset/split/resource checksums and actual guidance coverage.
- The checkpoint bundle checksum and checkpoint-selection rule (`best-res` or `best-acc`).
- Evaluation command, reference-file checksum, evaluator revision, generated captions, and raw score files.
- Paper table/row mapping, observed metrics, and differences in preprocessing or protocol.

Run training and evaluation on the prepared resources, compare the raw scores with the intended paper row, and document discrepancies before describing the release as reproduced.


## Environment changes in this source release

The unmodified archive dependency list is preserved in
[requirements-original.txt](requirements-original.txt); it is provenance, not an
installation recommendation. The candidate requirements retain the archive's
PyTorch, TorchVision, Transformers, Accelerate, CLIP commit, NumPy, and pandas
versions. TensorFlow GPU 2.9.0 is replaced by TensorFlow CPU 2.15.1 for the scalar
logging used in this code, avoiding the incompatible NumPy constraint in the
old package. Unused scikit-learn, SciPy, and OpenCV direct requirements are
removed. NLTK and small support libraries are pinned, and Python 3 COCO
evaluation packages are included. This is not a complete transitive dependency
lockfile and has not been installed and tested as a full GPU environment.

## Release validation performed

On 2026-10-28, all released Python sources parsed and all 27 guidance JSON files
loaded. Explicit boolean parsing and configurable resource paths were checked,
including invalid boolean rejection and a synthetic VAD-file lookup with
missing-file diagnostics. A small three-stage fusion forward/backward check
passed with finite outputs and gradients under local PyTorch 2.9.1 CPU.
This does not validate the candidate PyTorch 1.13.0 CUDA environment.

The core GPT-2, VAD encoder, and fusion implementation files remain byte-for-byte
identical to the supplied archive. No paper metric, convergence claim, or
end-to-end inference result is claimed. Run the lightweight checks with:

```bash
python scripts/check_source.py
```
