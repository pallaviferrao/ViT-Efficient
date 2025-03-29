from Layers.data import prepare_data
from transformers import ViTForImageClassification, ViTImageProcessor
from Layers.Attention.vit_finetune import ViTSelfAttention
from Layers.Attention.vit_finetune_route import ViTSelfAttentionRoute
import torch
import numpy as np

from torch import nn, optim
from tqdm.auto import tqdm
# Training loop
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        ViTSelfAttentionRoute.LOAD_BALANCING_LOSSES.clear()
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        moh_loss = sum(ViTSelfAttentionRoute.LOAD_BALANCING_LOSSES) / max(
            len(ViTSelfAttentionRoute.LOAD_BALANCING_LOSSES), 1)
        loss += moh_loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)



            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def print_modules(model, device, config=None, ):

    for name, module in reversed(model._modules.items()):
        # print(name, type(module))

        if name == 'encoder':
            config = model.config
        if len(list(module.children())) > 0:
            model._modules[name] = print_modules(module, device, config)

        if name == 'attention':
            print("Is instance", module)
            model._modules[name] = ViTSelfAttentionRoute(config).to(device)
            print("Is instance", name, model._modules[name])
    return model


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # CIFAR-10 has 10 classes
    num_classes = 10

    # Define hyperparameters
    batch_size = 4
    learning_rate = 2e-5
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader, classes = prepare_data(batch_size=4)
    metric_name = "accuracy"

    # Load pre-trained ViT model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # Important when changing the number of classes
    )

    print_modules(model, device)
    model.to(device)
    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Training and evaluation
    print(f"Training on {device}")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print("Model saved successfully!")


if __name__ == "__main__":
    main()