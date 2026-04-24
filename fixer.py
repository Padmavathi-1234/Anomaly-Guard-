import sys
import re

def add_stderr(text):
    result = []
    i = 0
    while i < len(text):
        m = re.search(r'\bprint\s*\(', text[i:])
        if not m:
            result.append(text[i:])
            break
        
        start_idx = i + m.start()
        result.append(text[i:start_idx])
        
        paren_count = 0
        j = start_idx + len(m.group())
        in_string = False
        string_char = ''
        escape = False

        while j < len(text):
            char = text[j]
            
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == string_char:
                    # Lookahead for triple quotes
                    if string_char in ('"', "'") and j+2 < len(text) and text[j:j+3] == string_char*3:
                        j += 2
                        in_string = False
                    elif string_char in ('"', "'"):
                        # single quote closed
                        in_string = False
            else:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                    if j+2 < len(text) and text[j:j+3] == char*3:
                        j += 2
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    if paren_count == 0:
                        break
                    paren_count -= 1
                    
            j += 1
            
        print_stmt = m.group() + text[start_idx + len(m.group()):j]
        
        if "flush=True" not in print_stmt and "file=sys.stderr" not in print_stmt:
            result.append(print_stmt + ", file=sys.stderr)")
        else:
            result.append(print_stmt + ")")
            
        i = j + 1
        
    return "".join(result)

path = "inference.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix HF_TOKEN
old_token_line = 'HF_TOKEN     = os.getenv("HF_TOKEN",     os.getenv("OPENAI_API_KEY", ""))'
new_token_lines = 'HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")\nif not HF_TOKEN:\n    raise ValueError("ERROR: HF_TOKEN or OPENAI_API_KEY environment variable is required")'
if old_token_line in text:
    text = text.replace(old_token_line, new_token_lines)

text = add_stderr(text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
    
print("Fixed successfully!")
