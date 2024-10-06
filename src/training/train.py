import os
import copy
import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration parameters for the model training."""
    img_height: int = 128
    img_width: int = 128
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    dataset_path: str = 'dataset'  
    model_save_path: str = 'best_model.pth'
    patience: int = 10
    fine_tune_epochs: int = 20
    num_workers: int = 4  # Set to 0 for debugging
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionModule(nn.Module):
    """
    Attention module to emphasize important features.
    """
    def __init__(self, in_channels: int):
        super(AttentionModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor instead.")
        
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiBranchModel(nn.Module):
    """
    Multi-branch neural network model for processing RGB and grayscale images.
    """
    def __init__(self, num_classes: int = 1):
        super(MultiBranchModel, self).__init__()
        # RGB branch
        self.rgb_branch = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.rgb_branch.classifier = nn.Identity()
        self.rgb_attention = AttentionModule(in_channels=1280)
        
        # Grayscale branch
        self.gray_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.gray_branch = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.gray_branch.classifier = nn.Identity()
        self.gray_attention = AttentionModule(in_channels=1280)
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(1280 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, rgb: torch.Tensor, gray: torch.Tensor) -> torch.Tensor:
        # RGB branch
        rgb_features = self.rgb_branch.features(rgb)
        rgb_features = self.rgb_attention(rgb_features)
        rgb_features = torch.mean(rgb_features, dim=[2, 3])
        
        # Grayscale branch
        gray = self.gray_conv(gray)
        gray_features = self.gray_branch.features(gray)
        gray_features = self.gray_attention(gray_features)
        gray_features = torch.mean(gray_features, dim=[2, 3])
        
        # Fusion
        combined = torch.cat((rgb_features, gray_features), dim=1)
        out = self.fusion(combined)
        return out.squeeze()


class PlantDataset(Dataset):
    """
    Custom dataset for plant images.
    """
    def __init__(self, file_paths: List[str], labels: List[float],
                 transform_rgb: transforms.Compose = None,
                 transform_gray: transforms.Compose = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform_rgb = transform_rgb
        self.transform_gray = transform_gray

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise e
        
        # Convert to grayscale
        gray_image = image.convert('L')
        
        # Apply transformations
        if self.transform_rgb:
            rgb = self.transform_rgb(image)
        if self.transform_gray:
            gray = self.transform_gray(gray_image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return rgb, gray, label


def get_class_labels(directory: str) -> List[str]:
    """
    Extract class labels from folder names in the dataset directory.
    """
    classes = []
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            if 'healthy' in folder.lower():
                classes.append('healthy')
            else:
                classes.append('disease')
    classes = sorted(list(set(classes)))
    return classes


def load_class_labels(dataset_path: str) -> List[str]:
    """
    Load and print class labels from the dataset.
    """
    classes = get_class_labels(dataset_path)
    logger.info(f"Detected classes: {classes}")
    return classes


def load_data(data_dir: str) -> Tuple[List[str], List[float]]:
    """
    Load image file paths and corresponding labels from the dataset directory.
    """
    file_paths = []
    labels = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(folder_path, img_file)
                file_paths.append(img_path)
                label = 0.0 if 'healthy' in folder.lower() else 1.0
                labels.append(label)
    return file_paths, labels


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for rgb, gray, labels in progress_bar:
        rgb = rgb.to(device)
        gray = gray.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb, gray)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * rgb.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = (preds == labels).float().mean().item()
        progress_bar.set_postfix({'Loss': loss.item(), 'Acc': acc})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for rgb, gray, labels in progress_bar:
            rgb = rgb.to(device)
            gray = gray.to(device)
            labels = labels.to(device)

            outputs = model(rgb, gray)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * rgb.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = (preds == labels).float().mean().item()
            progress_bar.set_postfix({'Loss': loss.item(), 'Acc': acc})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    """Main function to orchestrate training and evaluation."""
    config = Config()

    # Load class labels
    classes = load_class_labels(config.dataset_path)

    # Load data
    file_paths, labels = load_data(config.dataset_path)
    logger.info(f"Total samples: {len(file_paths)}")

    # Split data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    logger.info(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    # Define transformations
    transform_rgb = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((config.img_height, config.img_width), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    transform_gray = transforms.Compose([
        transforms.Resize((config.img_height, config.img_width)),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((config.img_height, config.img_width), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],  # Single channel mean
                             std=[0.229])   # Single channel std
    ])

    # Create datasets and dataloaders
    train_dataset = PlantDataset(train_paths, train_labels, transform_rgb, transform_gray)
    val_dataset = PlantDataset(val_paths, val_labels, transform_rgb, transform_gray)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    # Initialize model
    device = torch.device(config.device)
    model = MultiBranchModel(num_classes=1).to(device)
    logger.info(f"Using device: {device}")

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Callbacks and training state
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    # Training loop
    for epoch in range(1, config.epochs + 1):
        logger.info(f"Epoch {epoch}/{config.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}")

        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, config.model_save_path)
            logger.info("Saved best model.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"No improvement. Early stop counter: {early_stop_counter}/{config.patience}")
            if early_stop_counter >= config.patience:
                logger.info("Early stopping triggered.")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Fine-tuning: Unfreeze the last few layers
    logger.info("Starting fine-tuning...")
    for param in model.rgb_branch.parameters():
        param.requires_grad = False
    for param in model.gray_branch.parameters():
        param.requires_grad = False

    # Unfreeze the last 20 layers
    for layer in list(model.rgb_branch.children())[-20:]:
        for param in layer.parameters():
            param.requires_grad = True
    for layer in list(model.gray_branch.children())[-20:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Reinitialize optimizer to optimize only the unfrozen parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config.learning_rate / 10)

    # Fine-tuning loop
    fine_tune_total_epochs = config.epochs + config.fine_tune_epochs
    for epoch in range(config.epochs + 1, fine_tune_total_epochs + 1):
        logger.info(f"Fine-tuning Epoch {epoch}/{fine_tune_total_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}")

        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, config.model_save_path)
            logger.info("Saved best fine-tuned model.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"No improvement. Early stop counter: {early_stop_counter}/{config.patience}")
            if early_stop_counter >= config.patience:
                logger.info("Early stopping triggered during fine-tuning.")
                break

    # Load best model weights after fine-tuning
    model.load_state_dict(best_model_wts)

    # Save the final model
    torch.save(model.state_dict(), config.model_save_path)
    logger.info("Training complete. Model saved.")


if __name__ == '__main__':
    main()
