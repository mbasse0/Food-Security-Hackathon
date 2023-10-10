import streamlit as st
import numpy as np
import pandas as pd

import os    
# from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor

from datasets import load_metric
import torch

from PIL import Image
import requests
import csv

from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

from PIL import Image
import requests
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'



st.title('Silo Classifier')
data = st.file_uploader("Upload an Image", type=["png", "jpg"])

if data is not None:
    model_checkpoint = "google/vit-base-patch16-224" # pre-trained model from which to fine-tune

    label2id = {'rien': 0, 'silo': 1}
    id2label = {0: 'rien', 1: 'silo'}


    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    model.load_state_dict(torch.load("modelev3.pt"))
    model.eval()



    st.image(data)
    image = Image.open(data)
    encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")

    # prepare image for the model
    encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    st.write("Predicted class:", model.config.id2label[predicted_class_idx])