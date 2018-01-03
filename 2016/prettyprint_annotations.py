import json

with open('train_val_videodatainfo.json', 'r') as handle:
    parsed = json.load(handle)

print json.dumps(parsed, indent=4, sort_keys=True)
