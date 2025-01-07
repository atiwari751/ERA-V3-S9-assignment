import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from resnet_model import ResNet50
from utils import load_checkpoint
import ast

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50()
model = torch.nn.DataParallel(model)
model = model.to(device)

# Load the checkpoint
checkpoint_path = "checkpoint.pth"
model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class labels from the file
with open("imagenet1000_clsidx_to_labels.txt") as f:
    class_labels = ast.literal_eval(f.read())

# Define the prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    results = []
    for i in range(top5_prob.size(0)):
        class_index = top5_catid[i].item()
        class_label = class_labels.get(class_index, "Unknown")
        prob = top5_prob[i].item() * 100
        results.append(f"{class_label}: {prob:.2f}%")
    
    return "\n".join(results)

# Create the Gradio interface
iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="text", title="ResNet 50 Image Classifier")
iface.launch() 