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

测试指令：
```
    print(f'{tokenizer.bos_token=}')
    print(f'{tokenizer.sep_token=}')
    print(f'{tokenizer.eos_token=}')
    print(f'{tokenizer.unk_token=}')
    print(f'{tokenizer.bos_token_id=}')
    print(f'{tokenizer.sep_token_id=}')
    print(f'{tokenizer.eos_token_id=}')
    print(f'{tokenizer.all_special_tokens=}')
```
输出结果：
```
tokenizer.bos_token='<|endoftext|>'
tokenizer.sep_token='[SEP]'
tokenizer.eos_token='<|endoftext|>'
tokenizer.unk_token='<|endoftext|>'
tokenizer.bos_token_id=50256
tokenizer.sep_token_id=50257
tokenizer.eos_token_id=50256
tokenizer.all_special_tokens=['<|endoftext|>', '[SEP]', '<|endoftext|>']
```

由于加入了一个新的token，导致tokenizer的长度变成了50258，与之前50257不匹配。

最简单的解决方案：由于我用不到bos_token，而tokenizer在初始化的时候会自动启用bos_token，所以干脆直接把bos_token的内容改为[SEP]。
```
    tokenizer.bos_token = '</s>'
    tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token
    sep_token = tokenizer.bos_token
  ```
这样的解决方案局限性很明显，如果我需要bos_token呢？放一个以后可能会需要的[链接](https://stackoverflow.com/questions/73322462/how-to-add-all-standard-special-tokens-to-my-hugging-face-tokenizer-and-model)
