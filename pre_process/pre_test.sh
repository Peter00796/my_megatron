python /mnt/pengyanxin/my_megatron/tools/preprocess_data.py \
    --input /mnt/pengyanxin/data/webtext.test.jsonl \
    --json-keys text \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file /mnt/pengyanxin/gpt_vocab/gpt2-vocab.json \
    --merge-file /mnt/pengyanxin/gpt_vocab/gpt2-merges.txt \
    --output-prefix /mnt/pengyanxin/data/output/test \
    --workers 4 \
    --partitions 1 \
    --log-interval 1000
