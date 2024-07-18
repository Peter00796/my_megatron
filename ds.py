from datasets import load_dataset
from sklearn.model_selection import train_test_split
# Load the dataset
dataset = load_dataset("imdb")

# Prepare train and test sets
train_texts = [item['text'] for item in dataset['train']]
test_texts = [item['text'] for item in dataset['test']]
train_texts, valid_texts = train_test_split(train_texts, test_size=0.1, random_state=42)


# Write to files
with open('data/imdb_train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_texts))

with open('data/imdb_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_texts))

with open('data/imdb_valid.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_texts))