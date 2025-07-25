import json

file_path = '/workspace/open-r1/stats/output/Qwen2.5-1.5B-Instruct-GRPO-Math12k-no-thinkng/train_metrics.jsonl'

with open(file_path, 'r') as f:
    for line in f:
        if not line.strip():
            continue  # skip empty lines
        try:
            entry = json.loads(line)
            # Process the entry as needed, for example:
            print(f"Step: {entry.get('step')}, Rank: {entry.get('rank')}")
            print("Metrics:", entry.get('metrics', {}))
            import pdb; pdb.set_trace()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON line: {e}")
