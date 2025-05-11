from transformers import AutoModelForCausalLM, AutoTokenizer

AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="local_llama3_1b")
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="local_llama3_1b")
