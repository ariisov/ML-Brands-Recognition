import os
import pandas as pd
from PIL import Image
import torch
import clip
from sklearn.model_selection import train_test_split
import torch.optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from os import environ
from dotenv import loadenv
loadenv()

csv_file = environ['CSV_FILE']
data_path = environ['BRANDS_IMAGES']

# Checking the existence of the CSV file
if os.path.exists(csv_file):
    try:
        # Read CSV with omission of erroneous lines
        df = pd.read_csv(csv_file, sep=';', on_bad_lines='skip', encoding='utf-8')
        
        print(f"DataFrame:\n{df.head()}")

    except Exception as e:
        print(f"Ошибка при чтении CSV-файла: {e}")
else:
    print("CSV-файл не найден.")

df['image_path'] = df['image_path'].astype(str)

train_data, test_data = train_test_split(df, test_size=0.3)
val_data, test_data = train_test_split(test_data, test_size=0.5)

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def preprocess_image(image):
    return preprocess(image).unsqueeze(0).to(device)

def preprocess_text(text):
    return clip.tokenize([text]).to(device)

def get_features(data):
    images = [preprocess_image(Image.open(os.path.join(data_path, row['image_path']))) for _, row in data.iterrows()]
    texts = [preprocess_text(row['brand']) for _, row in data.iterrows()]
    
    image_features = torch.cat([model.encode_image(image).detach().cpu() for image in images])
    text_features = torch.cat([model.encode_text(text).detach().cpu() for text in texts])
    
    return image_features, text_features

train_image_features, train_text_features = get_features(train_data)

classifier = LogisticRegression()
classifier.fit(train_image_features.numpy(), train_data['brand'])

test_image_features, test_text_features = get_features(test_data)

# Predictions
predictions = classifier.predict(test_image_features.numpy())

# Performance evaluation
accuracy = accuracy_score(test_data['brand'], predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_data['brand'], predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
