from unsloth import FastLanguageModel

print("Testing Llama-3.2-1B...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    load_in_4bit=True,
    max_seq_length=4096
)

print("✅ Llama-3.2-1B loaded successfully!")
print("Model is ready!")