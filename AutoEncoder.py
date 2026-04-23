import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_layer_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_layer_size, input_size),
            nn.ReLU()
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded

class SoftMax(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftMax, self).__init__()
        self.classifier = nn.Linear(input_size, num_classes)
        # Note: We do NOT apply Softmax here because nn.CrossEntropyLoss() 
        # already includes Softmax + Log operations internally.
        # Adding Softmax manually would cause incorrect training

    def forward(self, x):
        return self.classifier(x)

class StackedAE(nn.Module):
    def __init__(self, ae1, ae2, classifier):
        super(StackedAE, self).__init__()
        self.encoder1 = ae1.encoder
        self.encoder2 = ae2.encoder
        self.classifier = classifier

    def forward(self, input):
        input = self.encoder1(input)
        input = self.encoder2(input)
        out = self.classifier(input)
        return out

def train_model(model, dataloader, validation_loader, loss_fn, optimizer, epochs, name, output_dir, reconstruction=True): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    val_loss_history = []        # Validation loss
    loss_history = []            # Store Training loss per epoch  
    val_acc_history = []         # Validation accuracy (classification only)
    train_acc_history = []       # Training accuracy   
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for batch in dataloader:
            inputs = batch[0].view(batch[0].size(0), -1).to(device)

            if not reconstruction:
                labels = batch[1].to(device)  

            # Forward pass
            outputs = model(inputs)
            
            # Loss
            if reconstruction:
                loss = loss_fn(outputs, inputs)   # Autoencoder
            else:
                loss = loss_fn(outputs, labels)  # Classification

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

             # Track training accuracy
            if not reconstruction:
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if not reconstruction:
            train_accuracy = train_correct / train_total
            train_acc_history.append(train_accuracy)

        # ----- Validation -----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_batch in validation_loader:
                val_inputs = val_batch[0].view(val_batch[0].size(0), -1).to(device)

                if not reconstruction:
                    val_labels = val_batch[1].to(device)

                val_outputs = model(val_inputs)

                # Loss
                if reconstruction:
                    loss = loss_fn(val_outputs, val_inputs)
                else:
                    loss = loss_fn(val_outputs, val_labels)
                val_loss += loss.item()

                # Accuracy
                if not reconstruction:
                    _, predicted = torch.max(val_outputs, 1)
                    val_correct += (predicted == val_labels).sum().item()
                    val_total += val_labels.size(0)

        avg_val_loss = val_loss / len(validation_loader)
        val_loss_history.append(avg_val_loss)

        if not reconstruction:
            val_accuracy = val_correct / val_total
            val_acc_history.append(val_accuracy)

    plt.clf()        
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Over Epochs {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_plot_{name}.png"))

    if not reconstruction:
        plt.figure()
        plt.plot(train_acc_history, label='Training Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Validation Accuracy Over Epochs ({name})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"val_accuracy_{name}.png"))

def check_test(model, test_loader, name, class_names, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            inputs = images.view(images.size(0), -1).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    total = len(all_labels)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plt.clf()
    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(output_dir, f"confusion_matrix_{name}.npy"), cm)
    
    # Compute per-class accuracy (precision for each actual class)
    per_class_accuracy = (cm.diagonal() / cm.sum(axis=1)) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)

    for i, acc in enumerate(per_class_accuracy):
        plt.text(cm.shape[1] + 0.5, i, f"{acc:.1f}%", va='center')
        
    plt.title(f"Confusion Matrix on Test Set - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name}.png"))

def load_data(dataset_type, data_dir):
    transform = transforms.Compose([
        transforms.Grayscale(),          
        transforms.Resize((48, 48)),     
        transforms.ToTensor()            
    ])

    # Load the entire dataset from the single provided directory
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    class_names = full_dataset.classes
    print(f"Classes loaded ({dataset_type} dataset):", class_names)
    
    train_size = int(0.7 * len(full_dataset))
    validation_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    
    train_dataset, validation_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, validation_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, validation_loader, class_names

def create_encoded_dataset(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    encoded_data = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            encoded = model.encoder(inputs)
            for i in range(encoded.size(0)):
                encoded_data.append((encoded[i].cpu(), labels[i]))
    return torch.utils.data.DataLoader(encoded_data, batch_size=64, shuffle=True)

def main():
    parser = argparse.ArgumentParser(description="Train an AutoEncoder setup.")
    parser.add_argument("--dataset_type", type=str, choices=['big', 'small'], required=True, help="Type of dataset to load ('big' or 'small').")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the main directory containing the dataset.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output plots and matrices.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_size = 48 * 48
    hidden_layer1_size = 100
    hidden_layer2_size = 50
    num_of_classes = 7
    criterion_AE = nn.MSELoss()
    criterion_classifier = nn.CrossEntropyLoss()
    num_epochs = 300

    train_loader, test_loader, validation_loader, class_names = load_data(args.dataset_type, args.data_dir)

    ae1 = AutoEncoder(input_size, hidden_layer1_size)
    ae2 = AutoEncoder(hidden_layer1_size, hidden_layer2_size)
    softmax = SoftMax(hidden_layer2_size, num_of_classes)

    optimizer1 = torch.optim.Adam(ae1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(ae2.parameters(), lr=0.001)
    optimizer3 = torch.optim.Adam(softmax.parameters(), lr=0.001)

    print("--- Training AE1 ---")
    train_model(ae1, train_loader, validation_loader, criterion_AE, optimizer1, num_epochs, "Ae1", args.output_dir, reconstruction=True)
    
    encoded1_output_loader = create_encoded_dataset(ae1, train_loader)
    encoded1_output_loader_validation = create_encoded_dataset(ae1, validation_loader)
    
    print("\n--- Training AE2 ---")
    train_model(ae2, encoded1_output_loader, encoded1_output_loader_validation, criterion_AE, optimizer2, num_epochs, "Ae2", args.output_dir, reconstruction=True)

    encoded2_output_loader = create_encoded_dataset(ae2, encoded1_output_loader)
    encoded2_output_loader_validation = create_encoded_dataset(ae2, encoded1_output_loader_validation)
    
    print("\n--- Training Softmax Classifier ---")
    train_model(softmax, encoded2_output_loader, encoded2_output_loader_validation, criterion_classifier, optimizer3, num_epochs, "Softmax", args.output_dir, reconstruction=False)

    stacked_model = StackedAE(ae1, ae2, softmax)
    optimizerStacked = torch.optim.Adam(stacked_model.parameters(), lr=0.001)
    
    #checking the results before stack
    check_test(stacked_model, test_loader, "before_stack", class_names, args.output_dir)

    #train the stacked model together
    print("\n--- Training Full Stacked Model ---")
    train_model(stacked_model, train_loader, validation_loader, criterion_classifier, optimizerStacked, num_epochs, "Stacked", args.output_dir, reconstruction=False)
    check_test(stacked_model, test_loader, "after_stack", class_names, args.output_dir)

if __name__ == "__main__":
    main()