import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies if missing
try:
    import onnx
except ImportError:
    print("Installing onnx...")
    os.system("pip install onnx")
try:
    import torchvision
    if torchvision.__version__ < '0.14.0':
        print("torchvision version too old. Installing latest...")
        os.system("pip install torchvision --upgrade")
except ImportError:
    print("Installing torchvision...")
    os.system("pip install torchvision")

# Define directories in Google Colab (adjust if running locally)
base_colab_path = r"/content/drive/MyDrive/Colab Notebooks/Euro"
results_dir = os.path.join(base_colab_path, 'results_pytorch_all')
stored_test_images_dir = os.path.join(results_dir, 'stored_test_images')
data_dir = os.path.join(base_colab_path, 'data')

# Create necessary directories
for dir_path in [base_colab_path, results_dir, stored_test_images_dir, data_dir]:
    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created/Verified directory: {dir_path}")
    except PermissionError:
        print(f"Permission denied for {dir_path}. Check Google Drive permissions.")
        exit(1)

# Verify write permissions
try:
    test_path = os.path.join(data_dir, 'test.txt')
    with open(test_path, 'w') as f:
        f.write("Test")
    os.remove(test_path)
    print(f"Write permission confirmed for {data_dir}")
except PermissionError:
    print(f"Permission denied for {data_dir}. Check Google Drive permissions.")
    exit(1)

# Global variables
num_epochs = 40
batch_size = 64
learning_rate = 0.0001  # Changed to 0.0001 for stability
random_seed = 42
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'River', 'Residential', 'SeaLake'
]

def load_data(batch_size=32, val_split=0.15, test_split=0.15, random_seed=42):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Check if dataset exists
    dataset_path = os.path.join(data_dir, 'eurosat')
    download_needed = not os.path.exists(dataset_path) or not os.listdir(dataset_path)

    if not download_needed:
        print(f"Dataset found at {dataset_path}. Using existing dataset.")
        try:
            full_dataset = EuroSAT(root=data_dir, download=False, transform=transform)
            print(f"Loaded dataset with {len(full_dataset)} images")
        except Exception as e:
            print(f"Failed to load existing dataset: {e}")
            download_needed = True

    if download_needed:
        print(f"Dataset not found or empty at {dataset_path}. Attempting to download...")
        try:
            full_dataset = EuroSAT(root=data_dir, download=True, transform=transform)
            print(f"Dataset downloaded to {dataset_path}")
        except Exception as e:
            print(f"Failed to download EuroSAT: {e}")
            print("Please download the dataset manually from https://archive.ics.uci.edu/ml/datasets/EuroSAT")
            print(f"Extract to {dataset_path} with structure: {dataset_path}/AnnualCrop, {dataset_path}/Forest, etc.")
            exit(1)

    # Verify dataset contents
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print(f"Dataset folder {dataset_path} is empty or missing. Please ensure the dataset is correctly downloaded.")
        exit(1)
    else:
        print(f"Dataset verified. Contents: {os.listdir(dataset_path)}")

    generator = torch.Generator().manual_seed(random_seed)
    n_total = len(full_dataset)
    n_val = int(val_split * n_total)
    n_test = int(test_split * n_total)
    n_train = n_total - n_val - n_test

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator)

    unnormalized_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    raw_dataset = EuroSAT(root=data_dir, download=False, transform=unnormalized_transform)

    test_indices = getattr(test_dataset, 'indices', None) or test_dataset._indices
    for idx in test_indices[:5]:  # Store only the first 5 test images for visualization
        img_tensor, label = raw_dataset[idx]
        class_name = class_names[label]
        class_folder = os.path.join(stored_test_images_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        save_path = os.path.join(class_folder, f"{idx}.png")
        save_image(img_tensor, save_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    return model

def visualize_predictions(images, labels, preds, epoch, phase):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denorm_images = images * std.to(images.device) + mean.to(images.device)
    denorm_images = torch.clamp(denorm_images, 0, 1)
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        ax = axes[i]
        ax.imshow(denorm_images[i].permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{phase}_predictions_epoch_{epoch+1}.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                images, labels = next(iter(train_loader))
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                visualize_predictions(images, labels, preds, epoch, phase="train")

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss_avg)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        # Step the scheduler
        scheduler.step(val_loss_avg)
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))

    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

        # Visualize predictions on a few test images
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        visualize_predictions(images, labels, preds, epoch=num_epochs, phase="test")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()

def save_model(model):
    torch.save(model.state_dict(), os.path.join(results_dir, 'efficientnet_eurosat.pth'))

def export_model_to_onnx(model, device):
    try:
        onnx_path = os.path.join(results_dir, 'efficientnet_eurosat.onnx')
        dummy_input = torch.randn(1, 3, 64, 64).to(device)
        torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
        print(f"Model exported to ONNX format at {onnx_path}")
    except Exception as e:
        print(f"Failed to export to ONNX: {e}")
        print("Continuing without ONNX export.")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = load_data(batch_size, random_seed=random_seed)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
    plot_losses(train_losses, val_losses)
    evaluate_model(model, test_loader, device)

    save_model(model)
    export_model_to_onnx(model, device)