import os
from fileUtils import txtManager
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from model import diseaseCNN
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

txtMan = txtManager()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

#DATASET INFO
dataset_path = './Dataset/cervello/training'
image_size = (256, 256)

if not os.path.exists(".\\classes.txt"):
    txtMan.createEmpty("classes.txt")
    for className in os.listdir(dataset_path):
        txtMan.writeText("classes.txt", f"{className}\n", True)
num_classes = len(txtMan.getLines(".\\classes.txt"))

#TRAINING INFO
batch_size = 64
learning_rate = 0.001
num_epochs = 50

def loadDataset():  
    x_data = []
    y_data = []
    labels = os.listdir(dataset_path)
    print(f"Found a total of: {len(labels)} classes")
    
    total_images = sum(len(os.listdir(os.path.join(dataset_path, label))) for label in labels)
    with tqdm(total=total_images, desc="Loading images") as pbar:
        for directory in labels:
            dir_path = os.path.join(dataset_path, directory)
            for imagePath in os.listdir(dir_path):
                image_path = os.path.join(dir_path, imagePath)
                try:
                    image = Image.open(image_path).convert('L') #grayscale = L
                    image = image.resize(image_size)
                    image_array = np.array(image, dtype=np.float32) / 255.0
                    image_array = np.expand_dims(image_array, axis=0) #add dim in order to be compatible with pytorch tensor
                    
                    x_data.append(image_array)
                    y_data.append(labels.index(directory))
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                finally:
                    pbar.update(1)
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    print(f"\nDataset loaded with {len(x_data)} images and {len(y_data)} labels")
    print(f"X_data shape: {x_data.shape}\nY_data shape: {y_data.shape}")
    return x_data, y_data

if __name__ == "__main__":
    x_data, y_data = loadDataset()
    x_data = torch.from_numpy(x_data).to(torch.float32)
    y_data = torch.from_numpy(y_data).to(torch.long)

    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    
    model = diseaseCNN(numClasses=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #graphs info
    epoch_losses = []  
    epoch_accuracies = []  
    all_preds = []  
    all_labels = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0 
        
        #for each batch
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # Add batch loss to total epoch loss
            
            # accuracy info
            _, predicted = torch.max(outputs, 1) #predicted is a tensor with all predicted labels of batch
            correct_preds += (predicted == labels).sum().item() #correct preds is sum of boolean tensor
            total_preds += labels.size(0)
            
            # Store pred and true labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_accuracy = correct_preds / total_preds
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.2f}, Accuracy: {epoch_accuracy*100:.2f}%')
    
    model_save_path = './brain_cnn_model.pth'
    torch.save(model.state_dict(), model_save_path)
    
    optimizer_save_path = './optimizer_state.pth'
    torch.save(optimizer.state_dict(), optimizer_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot loss and accuracy
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, num_epochs+1), epoch_losses, color='tab:red', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')  
    ax2.plot(range(1, num_epochs+1), epoch_accuracies, color='tab:blue', label='Training Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    fig.tight_layout()  
    plt.title('Training Loss and Accuracy Over Epochs')
    plt.show()
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, np.arange(num_classes))
    plt.yticks(tick_marks, np.arange(num_classes))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()