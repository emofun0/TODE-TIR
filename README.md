

# ECE450 Robust Robotic Manipulation of Transparent Objects under Perception Uncertainties

## Data Alignment
Single process:
```bash
python alignment.py single
```

Multi process:
```bash
python alignment.py multi
```

Processed data structure:
```plaintext
processed_seq_H_01/
├── 000001/
│   ├── 000001.npz
│   └── mask/
│       ├── 000001_0.png
│       └── ...
...
```

## Dataloader_test
```bash
python dataloader_test.py
```

## Train
Train on TransCG:
```bash
python train.py
```

Train on TransPose:
```bash
python train_transpose.py
```

## Test
Test TransCG pretrained model on TransCG:
```bash
python test.py
```

Test TransCG pretrained model on TransPose:
```bash
python test_transpose.py
```

## Inference
Inference with depth mask:
```bash
python sample_inference
```

Inference without depth mask:
```bash
python depth_inference.py single_test --config configs/depth_inference.yaml
```

## Finetune

Finetune TransCG pretrained model with TransPose:

```
python finetune.py
```























