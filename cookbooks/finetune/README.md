Fine-tune Qwen3-0.6B with self-cognition dataset

## Usage

1. Transform self-cognition dataset to mlx format

```bash
uv run modelscope download --dataset swift/self-cognition --local_dir ./datasets/self-cognition
uv run python cookbooks/finetune/transform.py --name AIR --author TronClass AIR
```

2. Download Qwen3-0.6B

```bash
uv run modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```

3. Fine-tune Qwen3-0.6B with self-cognition dataset

```bash
uv run mlx_lm.lora --config cookbooks/finetune/ft_qwen3_lora.yaml
```

4. Evaluate the fine-tuned model

```bash
uv run mlx_lm.chat --model ./models/Qwen3-0.6B --adapter-path cog_adapters
```

5. Start a local server
```bash
uv run mlx_lm.server --model ./models/Qwen3-0.6B --adapter-path cog_adapters --chat-template-args '{"enable_thinking":false}'
```

## Dataset

The dataset is from [self-cognition](../datasets/self-cognition/self_cognition.jsonl).