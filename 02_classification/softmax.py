import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Tekrarlanabilirlik için sabit bir seed belirleyelim.
torch.manual_seed(7)
# Genellikle parantez içine 42 sayısı konur. 
# Bunun sebebi Douglas Adams'ın "Otostopçunun Galaksi Rehberi" adlı bilim kurgu romanındaki bir espriye dayanır. Siz istediğiniz sayıyı verin.

num_samples_per_class = 100 # Her sınıf için örnek sayısı

# Sınıf 0: Merkez (-2, -2) etrafında
c0 = torch.randn(num_samples_per_class, 2) + torch.tensor([-2.0, -2.0])
y0 = torch.zeros(num_samples_per_class, dtype=torch.long)

# Sınıf 1: Merkez (2, -2) etrafında
c1 = torch.randn(num_samples_per_class, 2) + torch.tensor([2.0, -2.0])
y1 = torch.ones(num_samples_per_class, dtype=torch.long)

# Sınıf 2: Merkez (0, 2) etrafında
c2 = torch.randn(num_samples_per_class, 2) + torch.tensor([0.0, 2.0])
y2 = torch.full((num_samples_per_class,), 2, dtype=torch.long)

# Verileri birleştiriyoruz#
X = torch.cat([c0, c1, c2], dim=0)   
y = torch.cat([y0, y1, y2], dim=0)

# Mini-batch döngüsü için PyTorch DataLoader kuralım
dataset = TensorDataset(X, y)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model Mimarisi
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)         # [batch_size, 16]
        x = self.relu(x)        # [batch_size, 16]
        logits = self.fc2(x)    # [batch_size, 3]
        return logits

model = SimpleClassifier(input_dim=2, hidden_dim=16, num_classes=3)


# nn.CrossEntropyLoss kendi içinde Softmax ve NLLLoss birleşimini yapar.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


# Eğitim Döngüsü
epochs = 20

print("--- Eğitim Döngüsüne Hoşgeldiniz, Umarım Açık ve Anlaşılır Olur ---")
for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        # 1. Gradyan havuzunu sıfırla (Önceki adımdan kalan türevleri temizle) Bunu yapmazsanız gradyanlar birikir ve yanlış güncellemeler olur.
        optimizer.zero_grad()

        # 2. İleri Yayılım (Forward Pass): Modelden ham logitleri üret
        logits = model(X_batch)

        # 3. Kayıp Hesabı (Loss Calculation): Ham logit ve hedef etiket üzerinden
        loss = criterion(logits, y_batch)

        # 4. Geri Yayılım (Backpropagation): Zincir kuralı ile dL/dw gradyanlarını hesapla
        loss.backward()

        # 5. Ağırlık Güncellemesi: Ağırlıkları gradyan yönünün tersine güncelle
        optimizer.step()

        # Metrik takibi
        epoch_loss += loss.item() * X_batch.size(0)
        # Tahmin edilen sınıf en yüksek logit'e sahip indekstir. Tüm sınıfların olasılıklarının toplamı 1'e eşittir.
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    # Epoch sonu özet
    avg_loss = epoch_loss / total
    accuracy = (correct / total) * 100
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:02d}/{epochs}] -> Loss: {avg_loss:.4f} | Accuracy: %{accuracy:.2f}")


# Çıkarım / Tahmin (Inference & Olasılık İncelemesi)
print("\n--- Model Testi / Çıkarım Aşaması ---")
model.eval()

# Rastgele bir test noktası verelim: (-2.5, -1.8) -> Sınıf 0 merkezine çok yakın
test_point = torch.tensor([[-2.5, -1.8]])

with torch.no_grad():
    # 1. Modelden ham logit al
    raw_logits = model(test_point)
    
    # 2. Kullanıcıya/kendimize sunmak için Softmax ile olasılığa çevir
    probabilities = torch.softmax(raw_logits, dim=-1)
    
    # 3. En yüksek olasılığa sahip sınıfı seç
    predicted_class = torch.argmax(probabilities, dim=-1).item()

print(f"Test Noktası: {test_point.numpy().tolist()}")
print(f"Ham Logitler: {raw_logits.numpy().round(3).tolist()}")
print(f"Softmax Olasılık Dağılımı: {probabilities.numpy().round(4).tolist()}")
print(f"Seçilen Sınıf Tahmini: Sınıf {predicted_class}")