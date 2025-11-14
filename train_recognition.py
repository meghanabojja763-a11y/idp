# train_recognition.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/recognition', help='path to recognition dataset')
parser.add_argument('--epochs', type=int, default=12)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--out', default='models/recognition_alexnet.pth')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(args.data, transform=transform)
if len(dataset.classes) == 0:
    raise RuntimeError("No classes found in dataset. Create data/recognition/<PersonName>/ with images.")
loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds==labels).sum().item()
        total += imgs.size(0)
    print(f"Epoch {epoch+1}/{args.epochs} Loss:{running_loss/total:.4f} Acc:{correct/total:.4f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'classes': dataset.classes
}, args.out)
print("Saved:", args.out)
