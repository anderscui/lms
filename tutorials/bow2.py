# coding=utf-8
import re
import torch
import torch.nn as nn

torch.manual_seed(42)

data_raw = """Movies are fun for everyone. 1 Cinema
Watching movies is great fun. 1 Cinema
Enjoy a great movie today. 1 Cinema
Research is interesting and important. 3 Science
Learning math is very important. 3 Science
Science discovery is interesting. 3 Science
Rock is great to listen to. 2 Music
Listen to music for fun. 2 Music
Music is fun for everyone. 2 Music
Listen to folk music! 2 Music"""


def load_texts():
    lines = data_raw.splitlines()
    texts = []
    labels = []
    label_map = {}
    for line in lines:
        parts = line.rsplit(maxsplit=2)
        assert len(parts) == 3
        texts.append(parts[0])
        label_id, label_name = int(parts[-2]), parts[-1]
        labels.append(label_id)
        label_map[label_id] = label_name
    return texts, labels, label_map


def tokenize(text: str):
    return re.findall(r'\w+', text.lower())


def get_vocab(texts):
    tokens = {token for text in texts for token in tokenize(text)}
    return {token: i for i, token in enumerate(sorted(tokens))}


def doc_to_bow(doc: str, voc: dict):
    tokens = set(tokenize(doc))
    bow = [0] * len(voc)
    for token in tokens:
        if token in voc:
            bow[voc[token]] = 1
    return bow


docs, doc_labels, doc_label_map = load_texts()
num_labels = len(doc_label_map)
vocab = get_vocab(docs)
print(len(vocab), 'tokens loaded')

vectors = torch.tensor(
    [doc_to_bow(doc, vocab) for doc in docs],
    dtype=torch.float32
)
print(vectors.shape)

# use integer type, meet the `CrossEntropyLoss`
labels = torch.tensor(doc_labels, dtype=torch.long) - 1
label_names = [name for i, name in sorted(doc_label_map.items())]
print(labels.shape)
print('labels:', label_names)

input_dim = len(vocab)
hidden_dim = 50
output_dim = num_labels


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # full connected layer, from vocab to hidden size
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # activation, introduces non-linearity
        self.relu = nn.ReLU()
        # reduce intermediate to labels
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def predict(texts: list[str]):
    new_vecs = torch.tensor(
        [doc_to_bow(text, vocab) for text in texts],
        dtype=torch.float32
    )
    with torch.no_grad():
        outputs = model(new_vecs)
        print('raw outputs:', outputs)
        preds = torch.argmax(outputs, dim=1)
        print(preds.shape)
        print(preds)

    for i, text in enumerate(texts):
        print(f'{text}: {label_names[preds[i].item()]}')


model = SimpleClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for step in range(1000):
    optimizer.zero_grad()
    loss = criterion(model(vectors), labels)
    loss.backward()
    optimizer.step()

new_docs = ['Listening to rock is fun.', 'Learning math is very interesting.', 'Watching movies is great.']
predict(new_docs)
