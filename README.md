# CoTEL-D3X
Codes and datasets for CoTEL-D3X: A Chain-of-Thought Enhanced Large Language Model for Drug–Drug Interaction Triplet Extraction

## Methods
We proposed a Chain-of-Thought Enhanced Large Language Model for DDI Triplet Extraction (CoTEL-D3X). Based on the transformer architecture, we designed joint and pipeline methods that can perform end-to-end DDI triplet extraction in a generative manner. Our proposed approach builds upon the novel LLaMA series model as the foundation model and incorporates instruction tuning and Chain-of-Thought techniques to enhance the model’s understanding of task requirements and reasoning capabilities.

## Data
You need to get official data from here:
http://dx.doi.org/10.1016/j.jbi.2013.07.011

You need to (using LLaMA2 series model):   
① Request access from Meta:
https://www.llama.com/llama-downloads/   
② Apply for hf auth token:
https://huggingface.co/docs/hub/security-tokens/   

## Train
### Single GPU:

#### joint method
python qlora_finetune.py --data_path=train_joint.jsonl --output_path=ddi-joint --learning_rate=2e-4 --model_path=meta-llama/Llama-2-13b-hf --epochs=15 --max_length=1024 --micro_batch_size=8 --batch_size=16

#### pipeline method
python qlora_finetune.py --data_path=train_pipe.jsonl --output_path=ddi-pipe --learning_rate=2e-4 --model_path=meta-llama/Llama-2-13b-hf --epochs=15 --max_length=512 --micro_batch_size=8 --batch_size=16

### Multi GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12345 qlora_finetune.py --data_path=train_joint.jsonl --output_path=ddi-joint --model_path=meta-llama/Llama-2-13b-hf --epochs=15 --max_length=1024 --micro_batch_size=8 --batch_size=16

## Inference

#### joint method
python qlora_completion.py --data_path=test_joint.jsonl --output_path=prediction-ddi-pipe --lora_path=ddi-joint/checkpoint-xxx --model_path=meta-llama/Llama-2-13b-hf --max_new_tokens=1024

#### pipeline method
python qlora_completion.py --data_path=test_pipe.jsonl --output_path=prediction-ddi-pipe --lora_path=ddi-pipe/checkpoint-xxx --model_path=meta-llama/Llama-2-13b-hf --max_new_tokens=32

## References
TBD
CoTEL-D3X: A Chain-of-Thought Enhanced Large Language Model for Drug–Drug Interaction Triplet Extraction   
