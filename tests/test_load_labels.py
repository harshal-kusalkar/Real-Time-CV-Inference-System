import json 


with open("labels.json", "r") as f:
    labels = json.load(f)

print(labels)

# Access labels using index

print(labels['0'])