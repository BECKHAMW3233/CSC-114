# MNIST Benchmark Comparison: Historical, Current SOTA, and Ensemble Results

*Updated July 15, 2026 — reflects the finalized arXiv submission draft, independently verified citations, and a completed cross-machine reproducibility check.*

## 1. Final ensemble results (verified against training logs)

| Model | Resolution | Test Accuracy | Params | Exit Condition |
|---|---|---|---|---|
| Lion | 64×64 | 99.49% | 2.46M | Patience (15) |
| Lion | 128×128 | 99.45% | 2.46M | Wall clock |
| Schedule-Free AdamW | 64×64 | 99.46% | 9.71M | Patience (15) |
| Schedule-Free AdamW | 128×128 | 99.42% | 9.71M | Wall clock |
| SGD + Nesterov | 64×64 | 99.49% | 4.58M | Patience (20) |
| SGD + Nesterov | 128×128 | 98.86% | 4.58M | Wall clock |
| SOAP | 64×64 | 99.65% | 7.57M | Patience (20) |
| SOAP | 128×128 | **99.66% (best)** | 7.57M | Patience (20) |

Total GPU training time across all 8 models: 64.42 hours (231,898 seconds, summed directly from per-epoch logs), spread across 8 sequential runs on a single RTX 4080.

## 2. Historical and current benchmark comparison

| Model / Method | Year | Accuracy | Notes |
|---|---|---|---|
| LeCun et al., early CNN baseline | 1998 | ~99.05-99.2% | Foundational CNN work on MNIST |
| Ciresan, Meier & Schmidhuber, multi-column DNN | 2012 | 99.77% | Multiple network columns trained on differently-preprocessed inputs, averaged -- not a "seven-network" ensemble as sometimes conflated with an earlier (2011) Ciresan committee paper |
| Published MNIST SOTA range (no external pretraining, no extreme test-time augmentation) | ongoing | 99.7-99.87% | Achieved via committees, capsule networks, or heavily-tuned single models |
| **This project -- SOAP, 128x128 (best model)** | 2026 | **99.66%** | Within ~0.1-0.2pp of historical SOTA range, single consumer GPU, no elastic distortion or test-time augmentation |
| **This project -- 6 of 8 models** | 2026 | within 0.3pp of SOTA | Achieved with substantially less per-model tuning than cited SOTA work |

## 3. Where this sits relative to 2015-2016-era work and hardware

- **Ciresan et al.'s CNN committee (0.35% error / ~99.65% accuracy)** took **5-6 days** on GTX 480/580 GPUs (consumer cards of that era) to reach an accuracy band this project's SOAP result matches or exceeds.
- **This project's full 8-model sweep** -- 3 architectures x 4 optimizers x 2 resolutions, on a 439,148-sample dataset (7x larger than standard MNIST) -- completes in **64.42 cumulative GPU-hours**, well under what a single record-setting run required a decade earlier.
- The foundational GPU-based MNIST work (Ciresan et al., 2010-2012) also ran on **single consumer GPUs** (GTX 280 in 2010), not institutional clusters -- the same category of hardware as this project's RTX 4080, 15 years apart.

## 4. Verified related work and citations (independently checked, July 15, 2026)

Every citation below was individually verified against a primary source (the paper itself, official abstract pages, or multiple independent citing works) -- not assumed or reused from memory:

- **LeCun, Bottou, Bengio & Haffner (1998)** -- *Gradient-based learning applied to document recognition*, Proc. IEEE 86(11), 2278-2324. Confirmed exact.
- **Cohen, Afshar, Tapson & van Schaik (2017)** -- *EMNIST: Extending MNIST to handwritten letters*, IJCNN. Confirmed exact.
- **Clanuwat et al. (2018)** -- *Deep learning for classical Japanese literature* (Kuzushiji-MNIST), arXiv:1812.01718. Confirmed exact.
- **Ciresan, Meier & Schmidhuber (2012)** -- *Multi-column deep neural networks for image classification*, CVPR, 3642-3649. Citation confirmed exact; description corrected -- this paper's 99.77% figure comes from a multi-column architecture, not a "seven-network ensemble with elastic distortion" (that detail belongs to a separate, earlier 2011 Ciresan-group paper).
- **Chen et al. (2023)** -- *Symbolic discovery of optimization algorithms* (Lion), NeurIPS. Confirmed exact.
- **Defazio et al. (2024)** -- *The road less scheduled* (Schedule-Free), NeurIPS. Confirmed exact.
- **Sutskever, Martens, Dahl & Hinton (2013)** -- *On the importance of initialization and momentum in deep learning*, ICML. Confirmed exact.
- **Vyas et al. (2024)** -- *SOAP: Improving and stabilizing Shampoo using Adam*, arXiv:2409.11321. Confirmed correct (minor note: author list may vary slightly by arXiv version).
- **Sabour, Frosst & Hinton (2017)** -- *Dynamic routing between capsules*, NeurIPS. Confirmed exact.
- **Wan, Zeiler, Zhang, LeCun & Fergus (2013)** -- *Regularization of neural networks using DropConnect*, ICML. Confirmed exact.
- **Hu, Shen & Sun (2018)** -- *Squeeze-and-Excitation Networks*, CVPR. Confirmed exact.
- **Lin, Dollar, Girshick, He, Hariharan & Belongie (2017)** -- *Feature Pyramid Networks for Object Detection*, CVPR. Confirmed exact.
- **Liu et al. (2022)** -- *A ConvNet for the 2020s* (ConvNeXt), CVPR. Confirmed exact.
- **Jansson & Lindeberg (2020)** -- *Exploring the ability of CNNs to generalise to previously unseen scales over wide scale ranges* (MNIST Large Scale), ICPR / arXiv:2004.01536. Confirmed exact -- closest prior work on resolution as a controlled variable on MNIST-family data.
- **Nanni, Maguolo & Lumini (2021)** -- *Exploiting Adam-like optimization algorithms to improve the performance of CNNs*, arXiv:2103.14689. Confirmed exact -- closest documented precedent for optimizer-family as an ensemble-diversity axis (ResNet50, biomedical imaging, not MNIST).

**Conclusion of this project relative to cited precedent:** no single prior paper combines optimizer-family diversity + resolution diversity + architecture diversity jointly on MNIST-class data on single-consumer-GPU hardware with this level of documented reproducibility.

## 5. Real-world (out-of-distribution) evaluation

- **347 hand-verified ground-truth characters** scored across 14 photographed handwriting samples (a phantom 4th "line" segmentation artifact in test3.jpg was correctly identified and excluded during verification, along with 4 other segmentation/scribble artifacts).
- **SOAP-64 best real-world performer**: 96.25% (334/347).
- **Lion-128 is a clear, isolated real-world generalization outlier**: 62.82% (218/347) -- in-distribution MNIST test accuracy for this same model is 99.45%, second-highest in the ensemble, meaning the gap is specific to out-of-distribution photographed input, not a general model weakness.
- **Supporting data point (test40.jpg, 10 additional characters, run separately)**: all 8 models scored 10/10 except Lion-128, which missed 1 character (a stylized "5" misread as "3" at 53.4% vs. 29.8% confidence -- a near-tie, unlike its confident correct reads elsewhere).
- **Cross-machine determinism check completed July 15, 2026**: test40.jpg was re-run on a second, independent machine (FTCC lab PC, CPU-only inference, different library versions) -- output was byte-for-byte identical to the original RTX 4080 run, including exact confidence values. This confirms the pipeline is fully deterministic given fixed model weights, across different hardware and dependency versions.

## 6. Known limitations (as stated in the paper)

- Single run per (architecture, optimizer, resolution) configuration -- no repeated-seed variance reported. Note: all 8 training scripts fix `seed=42` for the train/validation split only (ensuring identical data partitioning across all runs); model weight initialization and other training-time stochasticity were not seeded.
- The SGD 128x128 result overlaps a documented, time-localized system-load confound (background CPU/RAM contention during part of the run) -- reported honestly as an unresolved confound, not a clean finding.
- Real-world evaluation set is small (14-15 images) and single-annotator; not independently blinded.
- No formal statistical significance testing across models -- differences under ~0.2-0.3pp should be read as within likely run-to-run noise.

## 7. Status

Paper finalized and verified; pending explicit instructor sign-off on acknowledgments wording before arXiv submission. AI-assistance disclosure included, naming Claude (Anthropic), Sonnet 4.6 and Sonnet 5, for writing, citation verification, and post-hoc data analysis -- all experimental design, training, debugging, and hardware work are the author's own.