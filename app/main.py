from flask import Flask, render_template, request
from base64 import b64decode
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
from . import inference

classes = ['அ', 'ஆ', 'ஓ', 'ஙூ', 'சூ', 'ஞூ', 'டூ', 'ணூ', 'தூ', 'நூ', 'பூ', 'மூ', 'யூ', 'ஃ', 'ரூ', 'லூ', 'வூ', 'ழூ', 'ளூ', 'றூ', 'னூ', 'ா', 'ெ', 'ே', 'க', 'ை', 'ஸ்ரீ', 'ஸு', 'ஷு', 'ஜு', 'ஹு', 'க்ஷு', 'ஸூ', 'ஷூ', 'ஜூ', 'ங', 'ஹூ', 'க்ஷூ', 'க்', 'ங்', 'ச்', 'ஞ்', 'ட்', 'ண்', 'த்', 'ந்', 'ச', 'ப்', 'ம்', 'ய்', 'ர்', 'ல்', 'வ்', 'ழ்', 'ள்', 'ற்', 'ன்', 'ஞ', 'ஸ்', 'ஷ்', 'ஜ்', 'ஹ்', 'க்ஷ்', 'ஔ', 'ட', 'ண', 'த', 'ந', 'இ', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன', 'ஈ', 'ஸ', 'ஷ', 'ஜ', 'ஹ', 'க்ஷ', 'கி', 'ஙி', 'சி', 'ஞி', 'டி', 'உ', 'ணி', 'தி', 'நி', 'பி', 'மி', 'யி', 'ரி', 'லி', 'வி', 'ழி', 'ஊ', 'ளி', 'றி', 'னி', 'ஸி', 'ஷி', 'ஜி', 'ஹி', 'க்ஷி', 'கீ', 'ஙீ', 'எ', 'சீ', 'ஞீ', 'டீ', 'ணீ', 'தீ', 'நீ', 'பீ', 'மீ', 'யீ', 'ரீ', 'ஏ', 'லீ', 'வீ', 'ழீ', 'ளீ', 'றீ', 'னீ', 'ஸீ', 'ஷீ', 'ஜீ', 'ஹீ', 'ஐ', 'க்ஷீ', 'கு', 'ஙு', 'சு', 'ஞு', 'டு', 'ணு', 'து', 'நு', 'பு', 'ஒ', 'மு', 'யு', 'ரு', 'லு', 'வு', 'ழு', 'ளு', 'று', 'னு', 'கூ']

app = Flask(__name__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1); self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1); self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1); self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1); self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1); self.bn6 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*8*8, 1024); self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512);    self.bn8 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 156)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x))); x = self.pool1(F.relu(self.bn6(self.conv6(x))))
        x = x.view(-1, 64*8*8)
        x = F.relu(self.bn7(self.fc1(x))); x = F.relu(self.bn8(self.fc2(x)))
        return F.softmax(self.fc3(x), dim=1)

net = Net()
net.load_state_dict(torch.load("tamil_net.pt", map_location=torch.device('cpu')))
net.eval()

my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def photo_to_tensor(file_stream):
    img = Image.open(file_stream)
    if img.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P': img = img.convert('RGBA')
        if img.mode in ('RGBA', 'LA'): bg.paste(img, mask=img.split()[-1])
        img = bg
    img = img.convert('L')
    if np.array(img).mean() > 128:
        img = ImageOps.invert(img)
    arr = np.array(img)
    threshold = max(arr.max() * 0.25, 30)
    arr = np.where(arr > threshold, arr, 0).astype(np.uint8)
    img = Image.fromarray(arr)
    bbox = img.getbbox()
    if not bbox: raise ValueError('Image appears blank.')
    padded = (max(0,bbox[0]-5), max(0,bbox[1]-5), min(img.width,bbox[2]+5), min(img.height,bbox[3]+5))
    img = img.crop(padded)
    img = img.filter(ImageFilter.MaxFilter(5))
    ratio = 48.0 / max(img.size)
    img = img.resize(tuple(int(round(x*ratio)) for x in img.size), Image.LANCZOS)
    arr = np.asarray(img)
    com = ndimage.center_of_mass(arr)
    result = Image.new('L', (64, 64))
    result.paste(img, (int(round(32.0 - com[1])), int(round(32.0 - com[0]))))
    return my_transforms(result).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    string_data = request.get_data().decode('utf-8')
    prediction = inference.get_prediction(string_data, net)
    return prediction

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'image' not in request.files: return 'No image uploaded', 400
    file = request.files['image']
    if not file.filename: return 'No file selected', 400
    try:
        tensor = photo_to_tensor(file.stream)
        with torch.no_grad():
            output = net(tensor)
            prob, predicted = torch.max(output.data, 1)
        char = classes[predicted.item()]
        confidence = int(round(prob.item() * 100))
        print(f"Upload: {char}  {confidence}%")
        return f"{char}|{confidence}"
    except ValueError as e:
        return str(e), 400
    except Exception as e:
        return f'Error: {str(e)}', 500

@app.route('/suggest', methods=['GET', 'POST'])
def suggest():
    return random.choice(classes)