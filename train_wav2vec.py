import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
from multiprocessing import cpu_count

# ---------------------------------
# Step 1: Dataset Class
# ---------------------------------
class JsonAudioDataset(Dataset):
    def __init__(self, json_file, processor, max_length=16000*30, replace_root_path=None):
        """
        Args:
            json_file: Path to the JSON file containing metadata.
            processor: Wav2Vec2 processor for feature extraction.
            max_length: Maximum audio length in samples (e.g., 30 seconds at 16kHz).
        """
        # Load the JSON data
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.processor = processor
        self.max_length = max_length
        self.replace_root_path = replace_root_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract metadata for the current entry
        entry = self.data[idx]
        audio_path = entry["audio_file_path"]
        start_time = entry["start_time"]
        end_time = entry["end_time"]
        label = entry["occupancy"]
        if self.replace_root_path:
            audio_path = self.replaceRootPath(audio_path, self.replace_root_path)

        # Load the audio file
        audio, sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)  # Convert to mono if stereo

        # Downsample by taking every 12th sample
        audio = audio[::12]
        sr = sr // 12  # Adjust the sampling rate accordingly (192kHz -> 16kHz)

        # Clip the audio between start_time and end_time
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio = audio[start_sample:end_sample]

        # Truncate or pad the audio to max_length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            padding = torch.zeros(self.max_length - len(audio))
            audio = torch.cat([audio, padding])

        # Process audio for Wav2Vec2
        inputs = self.processor(audio.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)

        # Return input values and label
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.float)

    def replaceRootPath(self, original_path, new_root):
        path_parts = original_path.split(os.sep)
        transformed_path = os.path.join(new_root, *path_parts[1:])
        return transformed_path

# ---------------------------------
# Step 2: Load Data and Prepare Dataset
# ---------------------------------
def load_datasets(train_json, val_json, processor):
    """
    Create training and validation datasets from JSON files.
    Args:
        train_json: Path to the training JSON file.
        val_json: Path to the validation JSON file.
        processor: Wav2Vec2 processor.
    """
    train_dataset = JsonAudioDataset(train_json, processor)
    val_dataset = JsonAudioDataset(val_json, processor)
    return train_dataset, val_dataset


# Load processor (tokenizer + feature extractor)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Paths to JSON files
train_json = "dataset_60-0s_train.json"
val_json = "dataset_60-0s_valid.json"

# Create Datasets and DataLoaders
num_workers = cpu_count()
train_dataset, val_dataset = load_datasets(train_json, val_json, processor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

# ---------------------------------
# Step 3: Model Definition
# ---------------------------------
# Load pre-trained Wav2Vec2 for regression
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=1,  # Regression task (1 output for predicting occupancy)
    problem_type="regression",
)

model.freeze_feature_extractor()  # Freeze feature extractor during fine-tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------------
# Step 4: Training Loop
# ---------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.MSELoss()  # Mean squared error for regression

def train(model, loader):
    model.train()
    train_loss = 0
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs).logits.squeeze(1)  # Predicted occupancy
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits.squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(loader)

# ---------------------------------
# Step 5: Run Training
# ---------------------------------
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train(model, train_loader)
    val_loss = evaluate(model, val_loader)
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
model.save_pretrained("wav2vec2-finetuned")
processor.save_pretrained("wav2vec2-finetuned")
