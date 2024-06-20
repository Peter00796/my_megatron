import json

file_path = '/mnt/pengyanxin/Megatron-LM/data/webtext.train.jsonl'

# Number of lines to display
num_lines = 1

with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i >= num_lines:
            break
        json_obj = json.loads(line)
        print(json.dumps(json_obj, indent=2))  # Pretty print JSON
