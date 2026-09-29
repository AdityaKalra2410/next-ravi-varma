# The Next Ravi Varma

Generating new paintings of scenes from Indian epics in the style of Raja Ravi Varma, by fine-tuning a Stable Diffusion model with LoRA.

## Team
| Member | Role |
|---|---|
| Member A | Data: collection, cleaning, captioning |
| Member B | Model: LoRA training on Stable Diffusion |
| Member C | Evaluation, demo, presentation |

## Plan

### Phase 1: Dataset (before midsem)
- [ ] Collect Ravi Varma paintings from Wikimedia Commons (public domain)
- [ ] Remove duplicates and non-paintings
- [ ] Separate oil paintings from printed copies (oleographs)
- [ ] Caption every painting with BLIP-2 + trigger word `rrvarma oil painting`
- [ ] Hold out a test set for evaluation

### Phase 2: Model (before midsem)
- [ ] Set up Stable Diffusion 1.5 on Colab/Kaggle GPU
- [ ] Train first LoRA on the oil paintings
- [ ] Log training loss, save checkpoints

### Phase 3: Evaluation (before midsem)
- [ ] Fixed set of 20 epic-scene test prompts
- [ ] Compare: base SD vs. base SD + "in the style of Raja Ravi Varma" vs. our LoRA
- [ ] Metrics: CLIPScore (text match), style similarity, KID, memorization check

### Phase 4: After midsem
- [ ] LLM-based prompt expansion
- [ ] ControlNet (OpenPose) for pose control
- [ ] Gradio web demo

## Timeline
| Week | Work |
|---|---|
| Week 1 | Dataset collection, cleaning, captioning; training setup |
| Week 2 | First LoRA training; baseline evaluation |
| Week 3 | Full evaluation; midsem flash talk |
| After midsem | Prompt expansion, ControlNet, web demo |

## Progress log
| Date | Update |
|---|---|
| | |
