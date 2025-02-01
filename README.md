## SILVAR - Reasoning Speech Instruction with Visual Language Model for Object Localization and Text Generation 🚀

<p  align="center"><img  src="./image/silvar.png"  width="700"></p>


**SILVAR** is an end-to-end visual language model that uses speech as input instructions for reasoning visual question answering and object localization.

<p  align="center"><img  src="./image/chatting.png"  width="700"></p>

### Supported Models:
SILVAR is designed with flexibility in mind, allowing seamless integration with various state-of-the-art models. Currently, the supported models are:

- **Language Models**: Mistral, Llama (2, 3, 3.1), Deepseek R1 (Distill Llama 8B)
- **Vision Encoders**: CLIP and its variants (e.g., Biomed-CLIP)
- **Audio Encoders**: Whisper (Tiny, Large)


## Installation

```bash
conda create -n silvar python=3.10.13
conda activate silvar
git clone https://github.com/Hanhpt23/SilVar.git
cd SilVar
pip install -r requirements.txt
```


## Training
### Visual encoder and audio encoder setting
We have released our checkpoint [here](https://drive.google.com/file/d/1flUkhhSJqA-jvzZABAgeIucHpu3WhBSv/view?usp=drive_link), you can download and use it as a pretrained weight or for inference.



### Training Configuration
- Set the pretrained checkpoint for downstream tasks [here](train_configs/train.yaml#L10) at Line 10.
- Set the training image path [here](train_configs/train.yaml#L35) at Line 35
- Set the training annotation path [here](train_configs/train.yaml#L36) at Line 36
- Set the training audio path [here](train_configs/train.yaml#L37) at Line 37
- Set the output directory [here](train_configs/train.yaml#L54) at Line 54
- Set the wandb token [here](train_configs/train.yaml#L69) at Line 69
- If you want to train the model end-to-end, set `freeze_vision` and `freeze_audio` to `False` [here](train_configs/train.yaml#L17) on lines 17 and 18


### Evaluation Configuration
- Set the checkpoint [here](eval_configs/evaluate.yaml#L10) at Line 10.
- Set the evaluation image path [here](eval_configs/evaluate.yaml#L36) at Line 36
- Set the evaluation annotation path [here](eval_configs/evaluate.yaml#L35) at Line 35
- Set the evaluation audio path [here](eval_configs/evaluate.yaml#L38) at Line 38
- Set the output directory [here](eval_configs/evaluate.yaml#L54) at Line 54

### Run
- To run on a terminal:

```bash
torchrun --nproc_per_node 2 train.py \
        --cfg-path train_configs/train.yaml\
        --cfg-eval-path eval_configs/evaluate.yaml\
        --eval-dataset audio_val
```

- To submit to an HPC:
```bash
sbatch scripts/silvar/train.sh
```

## Evaluation
- To run on a terminal:
```bash
torchrun --nproc_per_node 2 evaluate.py \
      --cfg-path eval_configs/evaluate.yaml\
      --eval-dataset audio_val
```

- To submit to an HPC:
```bash
sbatch scripts/silvar/evaluate.sh
```

## Dataset structure
```
Silvar
├── train
│   ├── audio
│   ├── images
│   ├── train.json
├── test
│   ├── audio
│   ├── images
│   ├── test.json

└── pretrained_checkpoint
    └── checkpoint_19.pth
```

#### Structure of `Silvar_sets.json`
```
[
      {
            "query": "",
            "outputs": "",
            "image": ""
      },
      ...
]
```



## Update
-  🔥 January 31, 2025 – We release SILVAR and its trained weights with [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) as the LLM.
-  🔥 December 20, 2024 – We release the open-source model and trained weights with the vision encoder [CLIP](https://huggingface.co/openai/clip-vit-base-patch16), the audio encoder [Whisper Tiny](https://huggingface.co/openai/whisper-tiny), and the LLM [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) .
- December 15, 2024 – We release the open-source model [here](https://anonymous.4open.science/r/SilVar-8911/README.md).