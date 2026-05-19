---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:90000
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: ce eval
      type: ce-eval
    metrics:
    - type: accuracy
      value: 0.8874
      name: Accuracy
    - type: accuracy_threshold
      value: 3.619974374771118
      name: Accuracy Threshold
    - type: f1
      value: 0.6988621997471556
      name: F1
    - type: f1_threshold
      value: -1.4393365383148193
      name: F1 Threshold
    - type: precision
      value: 0.6868787276341949
      name: Precision
    - type: recall
      value: 0.7112712300566135
      name: Recall
    - type: average_precision
      value: 0.7526066031778677
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 256 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['where is caldert county md', 'montgomery gansler s home county is an elusive key to gubernatorial primary victory douglas f gansler grew up in the leafy montgomery county community of chevy chase just north of the district and a short drive from the important federal jobs held by his father a high level defense department employee and many other county residents lowen s toy store and gifford s ice cream both in downtown bethesda were favorite destinations after college and law school gansler returned to the county raising a family just a few miles from his childhood home now gansler is trying to do something that no other montgomery county resident has done get elected governor of maryland he emphasizes this goal every time he appears in maryland s most populous county sometimes with his wife and mother in attendance we re going to have a governor from here who s going to stand up for the interests of montgomery county said gansler 51 as he opened a campaign office in rockville on a recent sunday the former county prosecutor now the state attorney general promised to stop what he called annapolis s practice of treating montgomery as the state s atm and to advocate for more state spending in the county which is increasingly diverse ethnically and economically to prevail in the june 24 democratic primary gansler has to win big in his home county most analysts say at this point he has a long way to go a washington post poll last month showed gansler ahead of his chief'],
    ['why did the house budget cuts irs', 'monument to a time when there was more optimism about the federal government since 2010 republicans on capitol hill have slashed the irs budget by 1 2 billion or about 17 percent adjusting for inflation just this fiscal year 346 million was cut by contrast cuts across the rest of the government have been far more modest and concentrated between 2012 and 2014 automatic spending reductions shrank non defense spending as adjusted for inflation by 1 3 percent while irs spending was chopped 5 6 percent according to scott lilly a budget expert at the center for american progress even in an era of shrinking government conservatives antipathy toward the tax collection agency stands out it is punishment for a string of missteps an extravagant conference for employees in anaheim calif the targeting of conservative groups seeking tax exemptions 1 million in bonuses given to agency employees who didn t pay their federal taxes we deliberately lowered irs funding to a level that will make the irs think twice about what you re doing and why you re doing it rep ander crenshaw r fla who chairs the house panel that sets the agency s budget said at a hearing last month the irs he said should focus on core missions providing taxpayer services leslie paige vice president for policy and communications at citizens against government waste said every government agency could sustain deep cuts and in particular the irs i think taxpayers and their representatives need to think long and hard about throwing more money'],
    ['what time of day do the washington shelters close', 'd c says it s out of options as it begins to turn homeless families away from shelters facing a surge in homelessness that has packed shelters to capacity for months the city has stopped guaranteeing a place to stay for homeless families that first sought help in february and march as a result at least 92 families were forced to search for safe spaces to sleep as temperatures warm up in the district the shift has alarmed advocates who fear that the lack of shelter will destabilize families already enduring a traumatic time they are turning their back on the public safety net that has helped to support these families said patricia mullahy fugere executive director for the washington legal clinic for the homeless it s going to be harder for families to stay in touch with caseworkers because they are struggling to figure out how to sleep at night and how to have a home base under a 2005 city law the district must provide homeless families with shelter whenever nighttime temperatures drop below freezing in years past the city has gone beyond its minimum legal obligation and continued to shelter families at the former d c general hospital campus and in motel rooms after hypothermia season while helping them seek other housing some homeless families remain in city provided shelter for a year or more but overwhelmed by a 135 percent increase in requests for family shelter this past winter the city resorted to placing families in makeshift shelters in recreation centers using'],
    ['what dunk did jan vesely throw', 'jan vesely s time ran out in washington jan vesely didn t know when or if he would ever experience a breakthrough with the wizards but he did know that time was running out the wizards put him on an uncomfortable clock last october when they declined his fourth year option and decided to send him into free agency a little sooner than expected within four months vesely discovered that the team that drafted him couldn t wait that long to get rid of him as washington dealt the 2011 sixth overall pick to the denver nuggets on thursday in exchange for andre miller his last contribution came in the final seconds of wednesday s 114 97 victory over the atlanta hawks coach randy wittman called on vesely to enter the game with 30 1 seconds remaining vesely moved slowly to scorer s table removed his warmups and adjusted his knee pads after otto porter jr missed a jumper vesely tapped the rebound back to garrett temple and adjusted his shorts as the clock melted down on the game and his 2½ year stint with the wizards his time with the wizards will be remembered for his draft night kiss his exciting dunks and his air ball free throws jan showed some signs he wasn t consistent enough for us wizards president ernie grunfeld said i think he has some abilities as far as athleticism and strength and running the floor and defensive abilities offensively he was very inconsistent for us a source close to vesely'],
    ['who was arrested causing cops to record him', 'police falsely told a man he couldn t film them i m an attorney he said i know what the law is one of the first things jesse bright did after being pulled over by police on a recent sunday afternoon was turn on his phone and begin filming bright was driving for uber to make some extra cash but he works full time as criminal defense attorney in north carolina as a lawyer he said he believes strongly that when people record their interactions with police it helps reduce confusion if their cases end up in court as he aimed his phone in the direction of officers and recorded bright was surprised to hear wilmington police sgt kenneth becker tell him that there was a new state law that prohibited him from recording police bright told the washington post that he knew better no such law exists in north carolina hey bud turn that off okay becker said no i ll keep recording thank you bright responded it s my right don t record me the police sergeant said you got me look bright said you re a police officer on duty i can record you be careful because there is a new law becker said turn it off or i ll take you to jail for recording you the video shows bright asking becker what is the law a tense exchange followed with becker telling bright to step out of his car calling him a jerk then warning him that he better hope officers'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'where is caldert county md',
    [
        'montgomery gansler s home county is an elusive key to gubernatorial primary victory douglas f gansler grew up in the leafy montgomery county community of chevy chase just north of the district and a short drive from the important federal jobs held by his father a high level defense department employee and many other county residents lowen s toy store and gifford s ice cream both in downtown bethesda were favorite destinations after college and law school gansler returned to the county raising a family just a few miles from his childhood home now gansler is trying to do something that no other montgomery county resident has done get elected governor of maryland he emphasizes this goal every time he appears in maryland s most populous county sometimes with his wife and mother in attendance we re going to have a governor from here who s going to stand up for the interests of montgomery county said gansler 51 as he opened a campaign office in rockville on a recent sunday the former county prosecutor now the state attorney general promised to stop what he called annapolis s practice of treating montgomery as the state s atm and to advocate for more state spending in the county which is increasingly diverse ethnically and economically to prevail in the june 24 democratic primary gansler has to win big in his home county most analysts say at this point he has a long way to go a washington post poll last month showed gansler ahead of his chief',
        'monument to a time when there was more optimism about the federal government since 2010 republicans on capitol hill have slashed the irs budget by 1 2 billion or about 17 percent adjusting for inflation just this fiscal year 346 million was cut by contrast cuts across the rest of the government have been far more modest and concentrated between 2012 and 2014 automatic spending reductions shrank non defense spending as adjusted for inflation by 1 3 percent while irs spending was chopped 5 6 percent according to scott lilly a budget expert at the center for american progress even in an era of shrinking government conservatives antipathy toward the tax collection agency stands out it is punishment for a string of missteps an extravagant conference for employees in anaheim calif the targeting of conservative groups seeking tax exemptions 1 million in bonuses given to agency employees who didn t pay their federal taxes we deliberately lowered irs funding to a level that will make the irs think twice about what you re doing and why you re doing it rep ander crenshaw r fla who chairs the house panel that sets the agency s budget said at a hearing last month the irs he said should focus on core missions providing taxpayer services leslie paige vice president for policy and communications at citizens against government waste said every government agency could sustain deep cuts and in particular the irs i think taxpayers and their representatives need to think long and hard about throwing more money',
        'd c says it s out of options as it begins to turn homeless families away from shelters facing a surge in homelessness that has packed shelters to capacity for months the city has stopped guaranteeing a place to stay for homeless families that first sought help in february and march as a result at least 92 families were forced to search for safe spaces to sleep as temperatures warm up in the district the shift has alarmed advocates who fear that the lack of shelter will destabilize families already enduring a traumatic time they are turning their back on the public safety net that has helped to support these families said patricia mullahy fugere executive director for the washington legal clinic for the homeless it s going to be harder for families to stay in touch with caseworkers because they are struggling to figure out how to sleep at night and how to have a home base under a 2005 city law the district must provide homeless families with shelter whenever nighttime temperatures drop below freezing in years past the city has gone beyond its minimum legal obligation and continued to shelter families at the former d c general hospital campus and in motel rooms after hypothermia season while helping them seek other housing some homeless families remain in city provided shelter for a year or more but overwhelmed by a 135 percent increase in requests for family shelter this past winter the city resorted to placing families in makeshift shelters in recreation centers using',
        'jan vesely s time ran out in washington jan vesely didn t know when or if he would ever experience a breakthrough with the wizards but he did know that time was running out the wizards put him on an uncomfortable clock last october when they declined his fourth year option and decided to send him into free agency a little sooner than expected within four months vesely discovered that the team that drafted him couldn t wait that long to get rid of him as washington dealt the 2011 sixth overall pick to the denver nuggets on thursday in exchange for andre miller his last contribution came in the final seconds of wednesday s 114 97 victory over the atlanta hawks coach randy wittman called on vesely to enter the game with 30 1 seconds remaining vesely moved slowly to scorer s table removed his warmups and adjusted his knee pads after otto porter jr missed a jumper vesely tapped the rebound back to garrett temple and adjusted his shorts as the clock melted down on the game and his 2½ year stint with the wizards his time with the wizards will be remembered for his draft night kiss his exciting dunks and his air ball free throws jan showed some signs he wasn t consistent enough for us wizards president ernie grunfeld said i think he has some abilities as far as athleticism and strength and running the floor and defensive abilities offensively he was very inconsistent for us a source close to vesely',
        'police falsely told a man he couldn t film them i m an attorney he said i know what the law is one of the first things jesse bright did after being pulled over by police on a recent sunday afternoon was turn on his phone and begin filming bright was driving for uber to make some extra cash but he works full time as criminal defense attorney in north carolina as a lawyer he said he believes strongly that when people record their interactions with police it helps reduce confusion if their cases end up in court as he aimed his phone in the direction of officers and recorded bright was surprised to hear wilmington police sgt kenneth becker tell him that there was a new state law that prohibited him from recording police bright told the washington post that he knew better no such law exists in north carolina hey bud turn that off okay becker said no i ll keep recording thank you bright responded it s my right don t record me the police sergeant said you got me look bright said you re a police officer on duty i can record you be careful because there is a new law becker said turn it off or i ll take you to jail for recording you the video shows bright asking becker what is the law a tense exchange followed with becker telling bright to step out of his car calling him a jerk then warning him that he better hope officers',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `ce-eval`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.8874     |
| accuracy_threshold    | 3.62       |
| f1                    | 0.6989     |
| f1_threshold          | -1.4393    |
| precision             | 0.6869     |
| recall                | 0.7113     |
| **average_precision** | **0.7526** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 90,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                    | sentence_1                                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                        | string                                                                                            | float                                                          |
  | details | <ul><li>min: 9 characters</li><li>mean: 38.12 characters</li><li>max: 98 characters</li></ul> | <ul><li>min: 4 characters</li><li>mean: 1186.98 characters</li><li>max: 1685 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.21</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>where is caldert county md</code>                        | <code>montgomery gansler s home county is an elusive key to gubernatorial primary victory douglas f gansler grew up in the leafy montgomery county community of chevy chase just north of the district and a short drive from the important federal jobs held by his father a high level defense department employee and many other county residents lowen s toy store and gifford s ice cream both in downtown bethesda were favorite destinations after college and law school gansler returned to the county raising a family just a few miles from his childhood home now gansler is trying to do something that no other montgomery county resident has done get elected governor of maryland he emphasizes this goal every time he appears in maryland s most populous county sometimes with his wife and mother in attendance we re going to have a governor from here who s going to stand up for the interests of montgomery county said gansler 51 as he opened a campaign office in rockville on a recent sunday the former county ...</code> | <code>0.0</code> |
  | <code>why did the house budget cuts irs</code>                 | <code>monument to a time when there was more optimism about the federal government since 2010 republicans on capitol hill have slashed the irs budget by 1 2 billion or about 17 percent adjusting for inflation just this fiscal year 346 million was cut by contrast cuts across the rest of the government have been far more modest and concentrated between 2012 and 2014 automatic spending reductions shrank non defense spending as adjusted for inflation by 1 3 percent while irs spending was chopped 5 6 percent according to scott lilly a budget expert at the center for american progress even in an era of shrinking government conservatives antipathy toward the tax collection agency stands out it is punishment for a string of missteps an extravagant conference for employees in anaheim calif the targeting of conservative groups seeking tax exemptions 1 million in bonuses given to agency employees who didn t pay their federal taxes we deliberately lowered irs funding to a level that will make the irs th...</code> | <code>0.0</code> |
  | <code>what time of day do the washington shelters close</code> | <code>d c says it s out of options as it begins to turn homeless families away from shelters facing a surge in homelessness that has packed shelters to capacity for months the city has stopped guaranteeing a place to stay for homeless families that first sought help in february and march as a result at least 92 families were forced to search for safe spaces to sleep as temperatures warm up in the district the shift has alarmed advocates who fear that the lack of shelter will destabilize families already enduring a traumatic time they are turning their back on the public safety net that has helped to support these families said patricia mullahy fugere executive director for the washington legal clinic for the homeless it s going to be harder for families to stay in touch with caseworkers because they are struggling to figure out how to sleep at night and how to have a home base under a 2005 city law the district must provide homeless families with shelter whenever nighttime temperatures drop ...</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss | ce-eval_average_precision |
|:------:|:-----:|:-------------:|:-------------------------:|
| 0.0222 | 500   | 0.9957        | -                         |
| 0.0444 | 1000  | 0.7521        | -                         |
| 0.0667 | 1500  | 0.6659        | -                         |
| 0.0889 | 2000  | 0.5842        | -                         |
| 0.1111 | 2500  | 0.5079        | -                         |
| 0.1333 | 3000  | 0.5087        | -                         |
| 0.1556 | 3500  | 0.4904        | -                         |
| 0.1778 | 4000  | 0.4855        | -                         |
| 0.2    | 4500  | 0.4804        | -                         |
| 0.2222 | 5000  | 0.487         | -                         |
| 0.2444 | 5500  | 0.4355        | -                         |
| 0.2667 | 6000  | 0.4489        | -                         |
| 0.2889 | 6500  | 0.4491        | -                         |
| 0.3111 | 7000  | 0.4683        | -                         |
| 0.3333 | 7500  | 0.4882        | -                         |
| 0.3556 | 8000  | 0.4214        | -                         |
| 0.3778 | 8500  | 0.4601        | -                         |
| 0.4    | 9000  | 0.4166        | -                         |
| 0.4222 | 9500  | 0.4452        | -                         |
| 0.4444 | 10000 | 0.4411        | -                         |
| 0.4667 | 10500 | 0.4517        | -                         |
| 0.4889 | 11000 | 0.4372        | -                         |
| 0.5111 | 11500 | 0.3864        | -                         |
| 0.5333 | 12000 | 0.4414        | -                         |
| 0.5556 | 12500 | 0.4373        | -                         |
| 0.5778 | 13000 | 0.4251        | -                         |
| 0.6    | 13500 | 0.4352        | -                         |
| 0.6222 | 14000 | 0.4296        | -                         |
| 0.6444 | 14500 | 0.3908        | -                         |
| 0.6667 | 15000 | 0.4233        | -                         |
| 0.6889 | 15500 | 0.3872        | -                         |
| 0.7111 | 16000 | 0.4541        | -                         |
| 0.7333 | 16500 | 0.4145        | -                         |
| 0.7556 | 17000 | 0.4495        | -                         |
| 0.7778 | 17500 | 0.4051        | -                         |
| 0.8    | 18000 | 0.4598        | -                         |
| 0.8222 | 18500 | 0.4209        | -                         |
| 0.8444 | 19000 | 0.4276        | -                         |
| 0.8667 | 19500 | 0.396         | -                         |
| 0.8889 | 20000 | 0.3969        | -                         |
| 0.9111 | 20500 | 0.4031        | -                         |
| 0.9333 | 21000 | 0.4359        | -                         |
| 0.9556 | 21500 | 0.4109        | -                         |
| 0.9778 | 22000 | 0.3988        | -                         |
| 1.0    | 22500 | 0.3959        | 0.7386                    |
| 1.0222 | 23000 | 0.37          | -                         |
| 1.0444 | 23500 | 0.348         | -                         |
| 1.0667 | 24000 | 0.3935        | -                         |
| 1.0889 | 24500 | 0.3785        | -                         |
| 1.1111 | 25000 | 0.3581        | -                         |
| 1.1333 | 25500 | 0.3757        | -                         |
| 1.1556 | 26000 | 0.382         | -                         |
| 1.1778 | 26500 | 0.3608        | -                         |
| 1.2    | 27000 | 0.3865        | -                         |
| 1.2222 | 27500 | 0.4061        | -                         |
| 1.2444 | 28000 | 0.369         | -                         |
| 1.2667 | 28500 | 0.3737        | -                         |
| 1.2889 | 29000 | 0.3994        | -                         |
| 1.3111 | 29500 | 0.375         | -                         |
| 1.3333 | 30000 | 0.3724        | -                         |
| 1.3556 | 30500 | 0.374         | -                         |
| 1.3778 | 31000 | 0.3601        | -                         |
| 1.4    | 31500 | 0.3477        | -                         |
| 1.4222 | 32000 | 0.3913        | -                         |
| 1.4444 | 32500 | 0.4003        | -                         |
| 1.4667 | 33000 | 0.3791        | -                         |
| 1.4889 | 33500 | 0.3509        | -                         |
| 1.5111 | 34000 | 0.3979        | -                         |
| 1.5333 | 34500 | 0.3897        | -                         |
| 1.5556 | 35000 | 0.3633        | -                         |
| 1.5778 | 35500 | 0.3652        | -                         |
| 1.6    | 36000 | 0.3332        | -                         |
| 1.6222 | 36500 | 0.3686        | -                         |
| 1.6444 | 37000 | 0.3494        | -                         |
| 1.6667 | 37500 | 0.3827        | -                         |
| 1.6889 | 38000 | 0.3592        | -                         |
| 1.7111 | 38500 | 0.375         | -                         |
| 1.7333 | 39000 | 0.3428        | -                         |
| 1.7556 | 39500 | 0.3837        | -                         |
| 1.7778 | 40000 | 0.3909        | -                         |
| 1.8    | 40500 | 0.3952        | -                         |
| 1.8222 | 41000 | 0.3728        | -                         |
| 1.8444 | 41500 | 0.3917        | -                         |
| 1.8667 | 42000 | 0.3934        | -                         |
| 1.8889 | 42500 | 0.37          | -                         |
| 1.9111 | 43000 | 0.3675        | -                         |
| 1.9333 | 43500 | 0.3482        | -                         |
| 1.9556 | 44000 | 0.3628        | -                         |
| 1.9778 | 44500 | 0.3642        | -                         |
| 2.0    | 45000 | 0.3621        | 0.7475                    |
| 2.0222 | 45500 | 0.3177        | -                         |
| 2.0444 | 46000 | 0.3088        | -                         |
| 2.0667 | 46500 | 0.3229        | -                         |
| 2.0889 | 47000 | 0.3296        | -                         |
| 2.1111 | 47500 | 0.3355        | -                         |
| 2.1333 | 48000 | 0.3154        | -                         |
| 2.1556 | 48500 | 0.3622        | -                         |
| 2.1778 | 49000 | 0.3079        | -                         |
| 2.2    | 49500 | 0.2995        | -                         |
| 2.2222 | 50000 | 0.3563        | -                         |
| 2.2444 | 50500 | 0.3192        | -                         |
| 2.2667 | 51000 | 0.3077        | -                         |
| 2.2889 | 51500 | 0.3215        | -                         |
| 2.3111 | 52000 | 0.2753        | -                         |
| 2.3333 | 52500 | 0.3524        | -                         |
| 2.3556 | 53000 | 0.3368        | -                         |
| 2.3778 | 53500 | 0.3358        | -                         |
| 2.4    | 54000 | 0.3467        | -                         |
| 2.4222 | 54500 | 0.3123        | -                         |
| 2.4444 | 55000 | 0.3303        | -                         |
| 2.4667 | 55500 | 0.3302        | -                         |
| 2.4889 | 56000 | 0.3302        | -                         |
| 2.5111 | 56500 | 0.3621        | -                         |
| 2.5333 | 57000 | 0.3277        | -                         |
| 2.5556 | 57500 | 0.3683        | -                         |
| 2.5778 | 58000 | 0.3221        | -                         |
| 2.6    | 58500 | 0.3706        | -                         |
| 2.6222 | 59000 | 0.331         | -                         |
| 2.6444 | 59500 | 0.3461        | -                         |
| 2.6667 | 60000 | 0.3226        | -                         |
| 2.6889 | 60500 | 0.3508        | -                         |
| 2.7111 | 61000 | 0.3481        | -                         |
| 2.7333 | 61500 | 0.3127        | -                         |
| 2.7556 | 62000 | 0.3135        | -                         |
| 2.7778 | 62500 | 0.2755        | -                         |
| 2.8    | 63000 | 0.337         | -                         |
| 2.8222 | 63500 | 0.3218        | -                         |
| 2.8444 | 64000 | 0.3388        | -                         |
| 2.8667 | 64500 | 0.318         | -                         |
| 2.8889 | 65000 | 0.3117        | -                         |
| 2.9111 | 65500 | 0.3169        | -                         |
| 2.9333 | 66000 | 0.3014        | -                         |
| 2.9556 | 66500 | 0.2932        | -                         |
| 2.9778 | 67000 | 0.3231        | -                         |
| 3.0    | 67500 | 0.3334        | 0.7526                    |

</details>

### Framework Versions
- Python: 3.10.6
- Sentence Transformers: 5.2.2
- Transformers: 4.44.2
- PyTorch: 2.7.1+cu118
- Accelerate: 1.13.0
- Datasets: 4.5.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->