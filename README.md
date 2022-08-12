## Pretrainedモデルの再事前学習の実装
 医療など特定の分野ではPretrainedモデルを該当ドメインのデータで再事前学習した方がいいのではないかという提案。
 元論文: Don't Stop Pretraining: [Adapt Language Models to Domains and Tasks, S.Gururangan et.al., ACL2020](https://arxiv.org/abs/2004.10964)

### 再事前学習時のTips from [Discussion of Kaggle AI4Code](https://www.kaggle.com/competitions/AI4Code/discussion/335294)
・Pre-train for as long as you can <br>
・More data is better than more epochs <br>
・Masking 15% of the tokens seems to be the sweet spot <br>
・Use an MLM probability schedule (e.g. start with 0.15 and decrease to 0.05 over the course of training)<br>
・Don't forget to use a warmup scheduler (linear warmup is fine)<br>
・Always use an optimizer with weight decay (e.g. AdamW)<br>
・If you are doing a question-answering task, you may want to use a different learning rate for the final linear layer (e.g. 0.01)<br>
・When fine-tuning, always use a small learning rate (e.g. 1e-5)<br>
・Don't forget to set the number of training epochs<br>
・If you are using a TPU, use gradient accumulation (e.g. accumulate_grad_batches=8)<br>
・If you are using a TPU, you may want to set a different batch size for each step (e.g. batch_size_per_step=batch_size_per_device * gradient_accumulation_steps)<br>
・For text classification, use DataCollatorForClassification<br>
・For question answering, use DataCollatorForQuestionAnswering<br>
・For sequence tagging, use DataCollatorForPermutationLanguageModeling<br>
・For next sentence prediction, use DataCollatorForNextSentencePrediction<br>
・For language modeling, use DataCollatorForLanguageModeling<br>
・For permutation language modeling use DataCollatorForPermutationLanguageModeling<br>

## References
[NBME 1st Solution](https://www.kaggle.com/code/currypurin/nbme-mlm/)<br>
[AI4Code Pairwise BertSmall Pretrain](https://www.kaggle.com/code/yuanzhezhou/ai4code-pairwise-bertsmall-pretrain/notebook)<br>
[An Overview of the Various BERT Pre-Training Methods](https://medium.com/analytics-vidhya/an-overview-of-the-various-bert-pre-training-methods-c365512342d8)<br>
Trainerを使う際の参考<br>
[huggingfaceのTrainerクラスを使えばFineTuningの学習コードがスッキリ書けてめちゃくちゃ便利です from Qiita@m__k](https://qiita.com/m__k/items/2c4e476d7ac81a3a44af)<br>
