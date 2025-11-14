import torch
from PIL import Image
import numpy as np
from model import diseaseCNN
from fileUtils import txtManager

tm = txtManager()

class predicter():
    def __init__(self):
        models_dict = {"lung_cnn":".\\Weights\\lung_cnn_model.pth", "brain_cnn":".\\Weights\\brain_cnn_model.pth"}
        models_classes_info = {"lung_cnn":".\\classes\\lung_cnn_model.txt", "brain_cnn":".\\classes\\brain_cnn_model.txt"}

        self.models_instances = {}
        self.models_classes = {}
        self.models_classNum = {}

        self.image_size = (256, 256)
        self.loadClasses(models_classes_info)
        self.loadModels(models_dict)

    def loadClasses(self, classes_dict):
        for key in classes_dict.keys():
            self.models_classes[key] = tm.getLines(classes_dict[key])
            self.models_classNum[key] = len(self.models_classes[key])
        
    def loadModels(self, models_dict):
        for key in models_dict.keys():
            self.models_instances[key] = diseaseCNN(numClasses=self.models_classNum[key])
            self.models_instances[key].load_state_dict(torch.load(models_dict[key], map_location=torch.device("cpu"), weights_only=True))

            
    def predict(self, image_path, model_id):
        try:
            image = Image.open(image_path).convert('L') #grayscale = L
            image = image.resize(self.image_size)
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).to(torch.float32).unsqueeze(0).unsqueeze(0) #[Batch, channels, height, width]
            image_tensor.to("cpu")
            
            with torch.no_grad():
                output = self.models_instances[model_id](image_tensor)
                _, predicted_class = torch.max(output, 1) #take the maximum of all classes prob. tensor, we only take index, acutal value is in _
                return self.models_classes[model_id][predicted_class.item()]
            
        except Exception as e:
            print(f"ERROR: {e}")
            
            
if __name__ == "__main__":
    testPred = predicter()
    print(testPred.predict(".\\Dataset\\test\\Tuberculosis\\Tuberculosis-557.jpg", "lung_cnn"))
