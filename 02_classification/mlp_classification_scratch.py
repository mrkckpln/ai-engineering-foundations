import torch


def relu(X):
    """Sıfırdan ReLU aktivasyonu: max(0, X)"""
    return torch.clamp(X, min=0.0)


def init_mlp_params(num_inputs, num_hiddens, num_outputs, scale=0.01):
    """Saf PyTorch tensörleri ile ağırlık ve bias ilklendirmesi."""
    W1 = torch.randn(num_inputs, num_hiddens, dtype=torch.float32) * scale
    b1 = torch.zeros(num_hiddens, dtype=torch.float32)

    W2 = torch.randn(num_hiddens, num_outputs, dtype=torch.float32) * scale
    b2 = torch.zeros(num_outputs, dtype=torch.float32)

    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_(True)

    return params


def mlp_forward(X, params):
    """İki katmanlı MLP ileri yayılım (Forward pass)."""
    W1, b1, W2, b2 = params
    # Gizli katman: H = ReLU(X * W1 + b1)
    H = relu(torch.matmul(X, W1) + b1)
    # Çıktı katmanı (Logits): O = H * W2 + b2
    logits = torch.matmul(H, W2) + b2
    return logits


def cross_entropy_loss(logits, targets):
    """Sayısal kararlı Cross-Entropy kaybı (Log-Sum-Exp trick)."""
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    stabilized_logits = logits - max_logits
    sum_exp = torch.sum(torch.exp(stabilized_logits), dim=1, keepdim=True)
    log_probs = stabilized_logits - torch.log(sum_exp)

    num_samples = logits.shape[0]
    loss = -torch.mean(log_probs[torch.arange(num_samples), targets])
    return loss


def sgd_step(params, lr, batch_size):
    """Sıfırdan Mini-batch Stochastic Gradient Descent (SGD)."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def compute_accuracy(logits, targets):
    """Modelin tahmin doğruluğunu (%) hesaplar."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return (correct / len(targets)) * 100.0


if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Sentetik Veri Kümesi Oluşturma (Spiral/Çoklu Küme mantığı)
    # 300 örnek, 4 öznitelik (feature), 3 sınıf
    num_samples = 300
    num_inputs = 4
    num_hiddens = 16
    num_outputs = 3

    X = torch.randn(num_samples, num_inputs, dtype=torch.float32)
    # Basit bir doğrusal olmayan etiketleme: normlara ve işaretlere göre sınıflama
    targets = (
        torch.sum(X**2, dim=1) > 3.5
    ).long() + (X[:, 0] > 0.0).long()
    targets = torch.clamp(targets, max=num_outputs - 1)

    # 2. Parametreleri İlklendir
    params = init_mlp_params(num_inputs, num_hiddens, num_outputs, scale=0.1)

    # 3. Eğitim Hiperparametreleri
    num_epochs = 100
    batch_size = 32
    lr = 0.5

    print(
        f"--- Sıfırdan MLP Eğitimi Başlıyor ({num_samples} örnek, {num_epochs} epoch) ---\n"
    )

    # 4. Eğitim Döngüsü
    for epoch in range(1, num_epochs + 1):
        indices = torch.randperm(num_samples)
        epoch_loss = 0.0
        num_batches = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            X_batch = X[batch_idx]
            y_batch = targets[batch_idx]

            # Forward
            logits = mlp_forward(X_batch, params)
            loss = cross_entropy_loss(logits, y_batch)

            # Backward
            loss.backward()

            # Optimizer Step
            sgd_step(params, lr=lr, batch_size=len(batch_idx))

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                total_logits = mlp_forward(X, params)
                acc = compute_accuracy(total_logits, targets)
                avg_loss = epoch_loss / num_batches
                print(
                    f"Epoch {epoch:03d} | Ortalama Kayıp: {avg_loss:.4f} | Doğruluk: %{acc:.1f}"
                )

    print("\n>> EĞİTİM TAMAMLANDI: Saf tensör MLP başarıyla yakınsadı.")