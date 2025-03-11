

from utils import load_experiment
from data import prepare_data
from torch import nn, optim
import torch
from vit import KVCacheAttention
from vit import FasterMultiHeadAttention



class TestClass():
    def __init__(self):
        self.batch_size = 32

    def print_modules(self, model, config):
        for name, module in reversed(model._modules.items()):
            # print(name, type(module))
            if len(list(module.children())) > 0:
                model._modules[name] = self.print_modules(module, config)
            if isinstance(module, FasterMultiHeadAttention):
                # print("Is instance", module)
                model._modules[name] = KVCacheAttention(config)
                # print("Is instance", name, model._modules[name])
        return model

    def print_modules2(self, model, config):
        for name, module in reversed(model._modules.items()):
            # print(name)
            if len(list(module.children())) > 0:
                self.print_modules2(module, config)
            if isinstance(module, MixtureOfAttention):
                print("Is instance", module)
        return model
    # @torch.no_grad()
    def test(self):
        batch_size = 32
        config, model, train_losses, test_losses, accuracies = load_experiment("vit-with-10-epochs")
        trainloader, testloader, _ = prepare_data(batch_size=batch_size)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
        total_loss = 0
        correct = 0
        model = self.print_modules(model,config)
        # self.print_modules2(model, config)
        print(len(testloader))
        # with torch.no_grad():
        for batch in testloader:
            batch = [t for t in batch]

            images, labels = batch

            # Get predictions

            # model(inputs)
            logits, _ = model(images, False)
            # Calculate the loss
            loss = loss_fn(logits, labels)

            # MixtureOfAttention.LOAD_BALANCING_LOSSES.clear()
            # batch = [t for t in batch]
            # images, labels = batch
            # # Zero the gradients
            # optimizer.zero_grad()
            # # Calculate the loss
            #
            # loss = loss_fn(model(images, True)[0], labels)
            # moh_loss = sum(MixtureOfAttention.LOAD_BALANCING_LOSSES) / max(
            #     len(MixtureOfAttention.LOAD_BALANCING_LOSSES), 1)
            # loss += moh_loss
            # # print("moh loss", moh_loss)
            # # Backpropagate the loss
            # loss.backward()
            #
            # # Update the model's parameters
            # optimizer.step()



            total_loss += loss.item() * len(images)
            # Calculate the accuracy
            # logits, _ = model(images, False)
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()

        print(f"Epoch:, Train loss: {total_loss:.4f}, Accuracy: {correct/len(testloader):.4f}")

def main():
    t= TestClass()
    t.test()
if __name__ == "__main__":
    main()
# @torch.no_grad()
# def evaluate(testloader):
#     with torch.no_grad():
#         for batch in testloader:
#             pass
#
# batch_size = 32
# config, model, train_losses, test_losses, accuracies = load_experiment("vit-with-10-epochs")
# trainloader, testloader, _ = prepare_data(batch_size=batch_size)
# loss_fn = nn.CrossEntropyLoss()
# total_loss = 0
# correct =0
# print(testloader)
# evaluate(testloader)

#
# with torch.no_grad():
#     model.eval()
#     for batch in testloader:
#         print(batch.shape)
#         pass
#     # Move the batch to the device
#     batch = [t for t in batch]
#     images, labels = batch
#
#     # Get predictions
#
#     # model(inputs)
#     logits, _ = model(images, False)
#
#     # Calculate the loss
#     loss = loss_fn(logits, labels)
#
#     total_loss += loss.item() * len(images)
#     # Calculate the accuracy
#     predictions = torch.argmax(logits, dim=1)
#     correct += torch.sum(predictions == labels).item()
#
# print(f"Epoch: {i+1}, Train loss: {total_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {correct/len(testloader):.4f}")