# Network-configuration-instruction-recommendation-based-on-pre-trained-language-model
My graduation project.

## Difficulties in coding
想要在训练集中加入分隔符[SEP]，和每句的结尾[EOS]，但是没有办法做到同时加入这两个token。
加入一个token的常规方法：

```
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir, add_prefix_space=False, ignore_mismatched_sizes=True)
    tokenizer.pad_token = tokenizer.eos_token 
```   
经过调查，选择了这样来加入两个token：
```
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
```
运行时报错：
```
RuntimeError: Error(s) in loading state_dict for GPT2LMHeadModel:
	size mismatch for transformer.wte.weight: copying a param with shape torch.Size([50257, 768]) from checkpoint, the shape in current model is torch.Size([50258, 768]).
	size mismatch for lm_head.weight: copying a param with shape torch.Size([50257, 768]) from checkpoint, the shape in current model is torch.Size([50258, 768]).
```
此时观察config.json，发现
```
  "bos_token_id": 50256,
  "eos_token_id": 50256,
```
怀疑是token的添加出现问题。
