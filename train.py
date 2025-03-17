import torch
from torch import nn, optim

from model.vit import ViTForClassfication
from Layers.utils import save_experiment, save_checkpoint
from Layers.data import prepare_data
# from vit import ViTForClassfication
# from vit import MixtureOfAttention
from Layers.Attention.moa_topk import MixtureOfAttention
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

# from torch.profiler import profile, record_function, ProfilerActivity

# from fvcore.nn import FlopCountAnalysis, parameter_count_table

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "intermediate_size": 4 * 48,  # 4 * hidden_size
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,  # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
    "attention-type":'multihead'
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        # warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)  # 5 epochs warmup
        # cosine = CosineAnnealingLR(optimizer, T_max=30)
        # self.scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i + 1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i + 1) % save_model_every_n_epochs == 0 and i + 1 != epochs:
                print('\tSave checkpoint at epoch', i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        # Save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        set_training_mode = True
        for batch in trainloader:
            # Move the batch to the device
            MixtureOfAttention.LOAD_BALANCING_LOSSES.clear()
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss

            loss = self.loss_fn(self.model(images, set_training_mode)[0], labels)
            if(config["attention-type"] == "moa"):
                moh_loss = sum(MixtureOfAttention.LOAD_BALANCING_LOSSES) / max(len(MixtureOfAttention.LOAD_BALANCING_LOSSES), 1)
                loss += moh_loss
                # print("moh loss", moh_loss)
            # Backpropagate the loss
            loss.backward()

            # Update the model's parameters
            self.optimizer.step()
            # self.scheduler.step()
            # MixtureOfAttention.LOAD_BALANCING_LOSSES.clear()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #     with record_function("model_inference"):
                    for batch in testloader:
                        # Move the batch to the device
                        batch = [t.to(self.device) for t in batch]
                        images, labels = batch

                        # Get predictions

                                # model(inputs)
                        logits, _ = self.model(images, False)

                        # flop_analysis = FlopCountAnalysis(self.model, input)
                        # print(flop_analysis)
                        # # print(f"Total FLOPs: {flop_analysis.total():.2e}")
                        #
                        # # Print parameter count
                        # print(parameter_count_table(self.model))
                                # Calculate the loss
                        loss = self.loss_fn(logits, labels)

                        total_loss += loss.item() * len(images)
                                # Calculate the accuracy
                        predictions = torch.argmax(logits, dim=1)
                        correct += torch.sum(predictions == labels).item()
                    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                    print("correct", correct)
                    print("accuracy", len(testloader.dataset))
                    accuracy = correct / len(testloader.dataset)
                    avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)
    parser.add_argument("--attention-type", type=str, default='moa')

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args



def main():
    args = parse_args()

    # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    config["attention-type"] = args.attention_type
    # Training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    # Load the CIFAR10 dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer
    model = ViTForClassfication(config, attention_type= args.attention_type)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)

    # with profile(activities=activities, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         model(inputs)
    #
    # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))


if __name__ == "__main__":
    main()