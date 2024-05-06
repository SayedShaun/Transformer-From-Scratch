import torch
from tqdm import tqdm
from dataloader import CombineDataset
from torch.utils.data import DataLoader
from torch import nn
from encoder_decoder_model import T5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CombineDataset(
    data_path="Data/english to bengali.csv",
    source_column="english_caption",
    target_column="bengali_caption"
)

dataloader = DataLoader(
    dataset, batch_size=64, shuffle=True,
    collate_fn=dataset.collate_fn
)

src_vocab = len(dataset.source.vocabs)
trg_vocab = len(dataset.target.vocabs)
hidden_size = 512
max_seq_len = 100
n_layers = 8
n_heads = 8
ff_hidden_size = 1024

model = T5(
    src_vocab,
    trg_vocab,
    hidden_size,
    max_seq_len,
    n_heads,
    ff_hidden_size,
    n_layers
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.target.vocabs["<PAD>"])
optimizer = torch.optim.Adam(model.parameters())


def train_fn(model, loss_fn, dataloader, optimizer, device):
    model.train()
    current_loss = 0.0
    for batch, (source, target) in enumerate(dataloader):
        source = source.to(device)
        target = target.to(device)
        src_mask = (source == dataset.source.vocabs["<PAD>"]).unsqueeze(1)
        print("src_mask shape", src_mask.shape)
        trg_mask = torch.tril(torch.ones((target.shape[1], target.shape[1]))).unsqueeze(0).to(device)

        optimizer.zero_grad()
        output = model(source, target, src_mask, trg_mask)
        output = output.flatten(0, 1)
        target = target.flatten()
        loss = loss_fn(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss += loss.item()
    avg_loss = current_loss / len(dataloader)
    return avg_loss


epochs = 10
loss_arr = []
for epoch in tqdm(range(epochs)):
    train_loss = train_fn(model, loss_fn, dataloader, optimizer, device)
    print("Train Loss", train_loss)
    loss_arr.append(train_loss)
