from unsloth import FastLanguageModel

print("Unsloth installed successfully!")

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct", 
    load_in_4bit=True
)

print("Model loaded successfully!")
print("Everything is working!")