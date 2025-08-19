import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tqdm import tqdm
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# ========================
# DATASET
# ========================
class SpeechT5Dataset(Dataset):
    def __init__(self, metadata_file, processor, sr=16000):
        self.data = pd.read_csv(metadata_file)
        self.processor = processor
        self.sr = sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        wav_path = row["path"].replace('\\', '/')
        text = row["text"]

        # Загружаем аудио
        audio, sr = librosa.load(wav_path, sr=self.sr)

        # Ограничение/нормализация
        max_length = self.sr * 25
        if len(audio) > max_length:
            audio = audio[:max_length]
        if len(audio) < self.sr // 2:
            audio = np.pad(audio, (0, self.sr // 2 - len(audio)), mode="constant")
        audio = audio / (np.max(np.abs(audio)) + 1e-9)

        # Текст в токены
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=600)

        # Превращаем в мел-спектрограмму
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        mel_spec = np.log(mel_spec + 1e-8).T   # (frames, n_mels)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(mel_spec, dtype=torch.float32),
            "text": text
        }

# ========================
# COLLATED BATCH
# ========================
def collate_fn(batch):
    # тексты
    max_text_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, attention_masks = [], []
    for x in batch:
        pad_len = max_text_len - x["input_ids"].shape[0]
        ids = torch.cat([x["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        mask = torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(ids)
        attention_masks.append(mask)

    # спектры
    max_spec_len = max(x["labels"].shape[0] for x in batch)
    n_mels = batch[0]["labels"].shape[1]
    labels, spec_masks = [], []
    for x in batch:
        spec = x["labels"]
        pad_len = max_spec_len - spec.shape[0]
        if pad_len > 0:
            spec = torch.cat([spec, torch.zeros(pad_len, n_mels)], dim=0)
        labels.append(spec)
        mask = torch.ones(spec.shape[0])
        mask[-pad_len:] = 0 if pad_len > 0 else 1
        spec_masks.append(mask)

    labels = torch.stack(labels)
    spec_masks = torch.stack(spec_masks)

    # подгоняем длину под reduction_factor
    rf = 2
    if labels.shape[1] % rf != 0:
        new_len = labels.shape[1] - (labels.shape[1] % rf)
        labels = labels[:, :new_len, :]
        spec_masks = spec_masks[:, :new_len]

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": labels,
        "decoder_attention_mask": spec_masks,
        "texts": [x["text"] for x in batch]
    }

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import os
from transformers import Wav2Vec2Model

# ========================
# LOSSES
# ========================
class MultiSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, y_pred, y_true):
        loss = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            S_pred = torch.stft(y_pred, n_fft=fft, hop_length=hop, win_length=win,
                                return_complex=True, center=True)
            S_true = torch.stft(y_true, n_fft=fft, hop_length=hop, win_length=win,
                                return_complex=True, center=True)

            loss_mag = F.l1_loss(torch.abs(S_pred), torch.abs(S_true))
            loss_phase = F.l1_loss(torch.angle(S_pred), torch.angle(S_true))
            loss += loss_mag + 0.1 * loss_phase
        return loss / len(self.fft_sizes)


class PerceptualLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, y_pred, y_true):
        # предполагаем, что вход — батч аудио (B, T)
        with torch.no_grad():
            feats_true = self.model(y_true).last_hidden_state
            feats_pred = self.model(y_pred).last_hidden_state
        return F.mse_loss(feats_pred, feats_true)


def train_speecht5(
    metadata="tts_metadata.csv",
    num_epochs=100,
    batch_size=8,
    lr=1e-4,
    w_base=1.0,
    w_mse=1.0,
    w_l1=0.5,
    w_sc=0.5,
    w_stft=0.5,
    w_perc=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", weights_only=False)
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", weights_only=False)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", weights_only=False)

    model.to(device)
    vocoder.to(device)

    dataset = SpeechT5Dataset(metadata, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    stft_loss_fn = MultiSTFTLoss().to(device)
    perceptual_loss_fn = PerceptualLoss(device=device)

    test_sentences = [
        "Привет, это тестовая речь.",
        "Многие деревья этого рода растут по несколько сотен лет.",
        "Общие затраты на Игры составят более полутора триллионов рублей."
    ]

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_base, total_mse, total_l1, total_sc, total_stft, total_perc = 0, 0, 0, 0, 0, 0, 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            decoder_mask = batch["decoder_attention_mask"].to(device)
            speaker_embeddings = torch.zeros(input_ids.shape[0], 512).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_mask,
                speaker_embeddings=speaker_embeddings
            )

            base_loss = outputs.loss

            min_len = min(outputs.spectrogram.size(1), labels.size(1))
            pred = outputs.spectrogram[:, :min_len, :]
            tgt = labels[:, :min_len, :]

            mse_loss = F.mse_loss(pred, tgt)
            l1_loss = F.l1_loss(pred, tgt)
            sc_loss = torch.norm(F.normalize(pred) - F.normalize(tgt), dim=-1).mean()

            with torch.no_grad():
                wav_true = vocoder(tgt)
                wav_pred = vocoder(pred)

            stft_l = stft_loss_fn(wav_pred.squeeze(1), wav_true.squeeze(1))
            perceptual_l = perceptual_loss_fn(wav_pred, wav_true)

            # комбинированный лосс с весами
            loss = (
                w_base * base_loss
                + w_mse * mse_loss
                + w_l1 * l1_loss
                + w_sc * sc_loss
                + w_stft * stft_l
                + w_perc * perceptual_l
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_base += base_loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            total_sc += sc_loss.item()
            total_stft += stft_l.item()
            total_perc += perceptual_l.item()

        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch {epoch+1}: total={avg_loss:.4f} | "
            f"base={total_base/len(dataloader):.4f} | "
            f"mse={total_mse/len(dataloader):.4f} | "
            f"l1={total_l1/len(dataloader):.4f} | "
            f"sc={total_sc/len(dataloader):.4f} | "
            f"stft={total_stft/len(dataloader):.4f} | "
            f"perc={total_perc/len(dataloader):.4f}"
        )

        if (epoch + 1) % 5 == 0:
            ckpt_path = f"checkpoints/checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved {ckpt_path}")

            # Тестовая генерация
            model.eval()
            out_dir = f"samples/epoch_{epoch+1}"
            os.makedirs(out_dir, exist_ok=True)
            for i, text in enumerate(test_sentences):
                inputs = processor(text=text, return_tensors="pt").to(device)
                speaker_embeddings = torch.zeros(1, 512).to(device)
                with torch.no_grad():
                    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
                    speech = vocoder(spectrogram)
                sf.write(f"{out_dir}/test_{i+1}.wav", speech.squeeze().cpu().numpy(), 16000)
                print(f"Saved {out_dir}/test_{i+1}.wav")

if __name__ == "__main__":
    train_speecht5()
