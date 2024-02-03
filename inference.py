import torch
import cv2
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import model
from PIL import Image

cure_js=json.load(open("cure.json","r",encoding='utf-8'))

id2class={0:"apple apple scab",
          1:"apple black rot",
          2:"apple cedar apple rust",
          3:"apple healthy"}

val_transform = A.Compose([
    A.Resize(height=256, width=256, p=1),
    A.Transpose(),
    A.Normalize(),
    ToTensorV2(),
])

def predict_image(file):
    image_path=file
    image=Image.open(image_path)
    image=np.array(image)
    #image=cv2.imread(image_path)
    #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=val_transform(image=image)["image"]
    image=torch.tensor(image)
    image=image.unsqueeze(0)
    with torch.no_grad():
        out=model(image)
    pred=torch.argmax(out)
    pred=pred.cpu().detach().numpy().item()
    disease=id2class[pred]
    solution=cure_js[disease]
    return {"disease":disease,"solution":solution}

