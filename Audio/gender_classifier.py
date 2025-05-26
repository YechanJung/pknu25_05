import torch

from voice_gender_classifier.model import ECAPA_gender

model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def classify_gender(file_path):
    with torch.no_grad():
        output = model.predict(file_path, device=device)
        return output
    

