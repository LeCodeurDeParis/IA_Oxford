import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


df = pd.read_csv("MVSA_Single/labelResultAll.txt", sep='\t')
df.reset_index(drop=True, inplace=True)
print(df.head())
print(df.columns)

df[['text', 'label']] = df['text,image'].str.split(',', expand=True)

df = df.drop(columns=['text,image'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    # Deleting mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Deleting hashtags (#hashtag) while keeping all words
    text = re.sub(r'#', '', text)
    # Deleting URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Deleting special symbols and ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Deleting numbers
    text = re.sub(r'\d+', '', text)
    # Deleting extra space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

data_folder = "MVSA_Single/data"
texts = []
image_paths = []

for idx in df['ID']:
    text_file = os.path.join(data_folder, f'{idx}.txt')
    image_file = os.path.join(data_folder, f'{idx}.jpg')

    if os.path.exists(text_file) and os.path.exists(image_file):
        with open(text_file, 'r', encoding='ISO-8859-1') as file:
            text_content = file.read().strip()
            text_content = preprocess_text(text_content)

        texts.append(text_content)
        image_paths.append(image_file)
    else:
        print(f"Warning: Missing files for ID {idx}")

df['text'] = texts
df['image_path'] = image_paths

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
print(df)

X_train_images, X_test_images, Y_train_images, Y_test_images = train_test_split(image_paths, df['label'], test_size=0.2, random_state=42, stratify=df['label'])
X_train_texts, X_test_texts, Y_train_texts, Y_test_texts = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Display length
print("Images :")
print("Training length : ", len(X_train_images), " Test length : ", len(X_test_images))
print("Labels of traing : ", len(Y_train_images), " Labels of test : ", len(Y_test_images))

print("Text :")
print("Training length : ", len(X_train_texts), " Test length : ", len(X_test_texts))
print("Labels of traing : ", len(Y_train_texts), " Labels of test : ", len(Y_test_texts))

vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_train_texts_transformed = vectorizer.fit_transform(X_train_texts)
X_test_texts_transformed = vectorizer.transform(X_test_texts)
vocab_size = len(vectorizer.vocabulary_)

print(X_train_texts_transformed.shape)
print(X_test_texts_transformed.shape)

# Convert the transformed data into tensor
def convert_to_tensor(X):
    return torch.tensor(X.toarray(), dtype=torch.long)

X_train_texts_tensor = convert_to_tensor(X_train_texts_transformed)
X_test_texts_tensor = convert_to_tensor(X_test_texts_transformed)

print(X_train_texts_tensor.shape)
print(X_test_texts_tensor.shape)

# Using the transformation during the Dataloaders creation

#Personalize Dataset for images
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels.values, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize every images to 128x128
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Creating DataLoaders for images
train_dataset_images = ImageDataset(X_train_images, Y_train_images, transform=transform)
test_dataset_images = ImageDataset(X_test_images, Y_test_images, transform=transform)

train_loader_images = DataLoader(train_dataset_images, batch_size=32, shuffle=True)
test_loader_images = DataLoader(test_dataset_images, batch_size=32, shuffle=False)

# Personalize Dataset for text
class TextDataset(Dataset):
    def __init__(self, texts_tensor, labels):
        self.texts_tensor = texts_tensor
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
         return self.texts_tensor[idx], self.labels.iloc[idx]

# Creating DataLoaders for texts
train_dataset_texts = TextDataset(X_train_texts_tensor, Y_train_texts)
test_dataset_texts = TextDataset(X_test_texts_tensor, Y_test_texts)

train_loader_texts = DataLoader(train_dataset_texts, batch_size=16, shuffle=True)
test_loader_texts = DataLoader(test_dataset_texts, batch_size=16, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)  # 3 résultats, neutre, positif, négatif

    def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      #print("After conv1 and pool:", x.size())
      x = self.pool(F.relu(self.conv2(x)))
      #print("After conv2 and pool:", x.size())
      x = self.pool(F.relu(self.conv3(x)))
      #print("After conv3 and pool:", x.size())
      x = x.view(-1, 64 * 16 * 16)  # Flatten
      #print("After flatten:", x.size())
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      #print("After fc layers:", x.size())
      return x

class LSTMTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(LSTMTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Taking the last output of the sequence
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Creating a CNN model instance
cnn_model = CNNModel()

# Loss function
criterion_cnn = nn.CrossEntropyLoss()

# Optimizer
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)

class TrainingCallback:
    def __init__(self, save_path='models', patience=5):
        self.save_path = save_path
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        os.makedirs(self.save_path, exist_ok=True)

    def on_epoch_end(self, model, epoch, val_loss):
        # Backup of the best model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            torch.save(model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
            print(f"Epoch {epoch+1}: Val loss improved to {val_loss:.4f}. Model saved.")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print("Early stopping due to no improvement.")
                return True  # Indicate that we should stop training

        return False  # Continue training
    
def train_cnn(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    callback = TrainingCallback(save_path='cnn_models', patience=3)
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            #print(f"Output size: {outputs.size()}")  # Devrait être (batch_size, num_classes)
            #print(f"Labels size: {labels.size()}")  # Devrait être (batch_size,)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)


        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)


        # Validation loss
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}")

        # Check if we should stop training
        if callback.on_epoch_end(model, epoch, epoch_val_loss):
            break
    return train_losses, val_losses, train_accuracy, val_accuracy

train_losses, val_losses, train_accuracy, val_accuracy = train_cnn(cnn_model, train_loader_images, test_loader_images, criterion_cnn, optimizer_cnn, device, num_epochs=20)

# Plot Loss
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


#Testing the model
def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
class_names = ['Negative', 'Neutral', 'Positive']
labels, preds = evaluate_model(cnn_model, test_loader_images, device)
plot_confusion_matrix(labels, preds, class_names)

# Initialising the model


# # Creating a LSTM model instance
lstm_model = LSTMTextModel(vocab_size=vocab_size, embed_size=100, hidden_size=128, num_classes=3)

vocab_size = 10000  # Size of vocabulary
embed_size = 128    # Size of  embeddings
hidden_size = 64    # Size of the LSTM's hidden states
num_classes = 3     # Number of class output
model = LSTMTextModel(vocab_size, embed_size, hidden_size, num_classes)

# Loss function
criterion_lstm = nn.CrossEntropyLoss()

# Optimizer
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

def train_lstm(model, train_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    train_losses = []
    train_accuracy = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i , (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)}")
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}")

    return train_losses, train_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses, train_accuracy = train_lstm(model, train_loader_texts, criterion_lstm, optimizer_lstm, device, num_epochs=10)

# Plot the Results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

#Confusion Matrix
class_names = ['Negative', 'Neutral', 'Positive']
labels, preds = evaluate_model(model, test_loader_texts, device)
plot_confusion_matrix(labels, preds, class_names)

class CombinedModel(nn.Module):
    def __init__(self, cnn_model, lstm_model, num_classes):
        super(CombinedModel, self).__init__()
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model

        # Combined fully connected layers
        self.fc1 = nn.Linear(512 + 128, 128)  # 512 from CNN's penultimate layer, 128 from LSTM's penultimate layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, image, text):
        # Forward pass through each model
        cnn_output = self.cnn_model.pool(F.relu(self.cnn_model.conv1(image)))
        cnn_output = self.cnn_model.pool(F.relu(self.cnn_model.conv2(cnn_output)))
        cnn_output = self.cnn_model.pool(F.relu(self.cnn_model.conv3(cnn_output)))
        cnn_output = cnn_output.view(-1, 64 * 16 * 16)
        cnn_output = F.relu(self.cnn_model.fc1(cnn_output))
        cnn_output = F.relu(self.cnn_model.fc2(cnn_output))

        lstm_output = self.lstm_model(text)

        # Concatenate the outputs
        combined_output = torch.cat((cnn_output, lstm_output), dim=1)

        # Forward pass through combined FC layers
        x = F.relu(self.fc1(combined_output))
        x = self.fc2(x)
        return x

# Initialise the combined model
combined_model = CombinedModel(cnn_model, lstm_model, num_classes=3)

def train_combined_model(model, cnn_loader, lstm_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for (images, labels_img), (texts, labels_txt) in zip(cnn_loader, lstm_loader):
            images, labels_img = images.to(device), labels_img.to(device)
            texts, labels_txt = texts.to(device), labels_txt.to(device)

            optimizer.zero_grad()

            outputs = model(images, texts)
            loss = criterion(outputs, labels_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels_img).sum().item()
            total_train += labels_img.size(0)

        epoch_train_loss = running_loss / len(cnn_loader.dataset)
        epoch_train_accuracy = correct_train / total_train

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}")
        
def evaluate_combined_model(model, cnn_loader, lstm_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for (images, labels_img), (texts, labels_txt) in zip(cnn_loader, lstm_loader):
            images, labels_img = images.to(device), labels_img.to(device)
            texts, labels_txt = texts.to(device), labels_txt.to(device)

            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_img).sum().item()
            total += labels_img.size(0)
            all_labels.extend(labels_img.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_labels, all_preds

def plot_confusion_matrix(labels, preds, class_names, title):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
    
cnn_accuracy, cnn_labels, cnn_preds = evaluate_combined_model(cnn_model, test_loader_images, test_loader_texts, device)
lstm_accuracy, lstm_labels, lstm_preds = evaluate_combined_model(lstm_model, test_loader_images, test_loader_texts, device)
combined_accuracy, combined_labels, combined_preds = evaluate_combined_model(combined_model, test_loader_images, test_loader_texts, device)

print(f"CNN Accuracy: {cnn_accuracy}")
print(f"LSTM Accuracy: {lstm_accuracy}")
print(f"Combined Model Accuracy: {combined_accuracy}")

class_names = ['Negative', 'Neutral', 'Positive']
plot_confusion_matrix(cnn_labels, cnn_preds, class_names, "Confusion Matrix for CNN")
plot_confusion_matrix(lstm_labels, lstm_preds, class_names, "Confusion Matrix for LSTM")
plot_confusion_matrix(combined_labels, combined_preds, class_names, "Confusion Matrix for Combined Model")