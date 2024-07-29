# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

hf_token = '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'
login(token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")

# Define the directory to save the checkpoint
save_directory = '/mnt/pengyanxin/my_megatron/examples/gpt3/checkpoints/gpt_xl_hf'

# Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
