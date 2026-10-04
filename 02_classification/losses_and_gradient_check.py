import torch


def cross_entropy_with_l2(logits, targets, weights, l2_lambda=0.0):
    """Sayısal kararlı (Log-Sum-Exp) Cross-Entropy kaybı ve L2 regülerizasyonu.

    logits: (N, C) boyutunda ham model çıktıları
    targets: (N,) boyutunda sınıf indeksleri (0 <= targets < C)
    weights: Modelin ağırlık tensörlerinin listesi [W1, W2, ...]
    l2_lambda: L2 ceza katsayısı
    """
    # 1. Log-Sum-Exp Hilesi ile Sayısal Kararlılık
    # Logitlerden satır bazlı maksimumu çıkararak exp() taşmasını (overflow) önlüyoruz
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    stabilized_logits = logits - max_logits

    # Softmax paydası ve log olasılıklar
    exp_logits = torch.exp(stabilized_logits)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    log_probs = stabilized_logits - torch.log(sum_exp)

    # 2. Doğru Sınıfların Log Olasılıklarını Toplama (NLL Loss)
    num_samples = logits.shape[0]
    correct_class_log_probs = log_probs[torch.arange(num_samples), targets]
    loss_ce = -torch.mean(correct_class_log_probs)

    # 3. L2 Regülerizasyonu (Weight Decay): 0.5 * lambda * sum(||W||^2)
    l2_loss = 0.0
    if l2_lambda > 0.0 and weights is not None:
        for w in weights:
            l2_loss = l2_loss + 0.5 * l2_lambda * torch.sum(w * w)

    total_loss = loss_ce + l2_loss
    return total_loss, log_probs


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    Analitik gradyanları sonlu farklar yöntemiyle (finite differences) doğrular.
    """
    print(f"\n--- Gradyan Doğrulama Başlatılıyor (Örneklem: {num_checks}) ---")
    passed = True

    for _ in range(num_checks):
        idx = tuple(torch.randint(0, dim_size, (1,)).item() for dim_size in x.shape)
        old_val = x[idx].item()

        # In-place güncellemeleri autograd takibinden çıkarıyoruz
        with torch.no_grad():
            # f(x + h)
            x[idx] = old_val + h
        loss_plus, _ = f()

        with torch.no_grad():
            # f(x - h)
            x[idx] = old_val - h
        loss_minus, _ = f()

        with torch.no_grad():
            # Orijinal değeri geri yükle
            x[idx] = old_val

        # Sayısal gradyan: [f(x + h) - f(x - h)] / (2 * h)
        numeric_grad = (loss_plus.item() - loss_minus.item()) / (2.0 * h)
        grad_analytic = analytic_grad[idx].item()

        # Bağıl hata (Relative Error)
        denom = max(abs(numeric_grad) + abs(grad_analytic), 1e-8)
        rel_error = abs(numeric_grad - grad_analytic) / denom

        print(f"İndeks: {str(idx):<12} | Sayısal: {numeric_grad:+.8f} | Analitik: {grad_analytic:+.8f} | Bağıl Hata: {rel_error:.2e}")

        if rel_error > 1e-5:
            passed = False

    if passed:
        print(">> SONUÇ: Gradyan doğrulama BAŞARILI! (Bağıl hata < 1e-5)\n")
    else:
        print(">> SONUÇ: DİKKAT! Yüksek bağıl hata tespit edildi.\n")


if __name__ == "__main__":
    torch.manual_seed(42)

    # Test verisi: 4 örnek, 3 sınıf
    N, C = 4, 3
    logits = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 2, 1, 0], dtype=torch.long)

    # Örnek ağırlık matrisi (L2 testi için)
    W = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    l2_lambda = 0.1

    # 1. Forward ve Autograd (Analitik Gradyan)
    loss, _ = cross_entropy_with_l2(logits, targets, [W], l2_lambda=l2_lambda)
    loss.backward()

    # 2. Logits üzerinde gradyan kontrolü
    def f_logits():
        return cross_entropy_with_l2(
            logits, targets, [W], l2_lambda=l2_lambda
        )

    print("Logits için Grad Check:")
    grad_check_sparse(f_logits, logits, logits.grad, num_checks=5)

    # 3. Ağırlık matrisi (W) üzerinde gradyan kontrolü
    def f_weights():
        return cross_entropy_with_l2(
            logits, targets, [W], l2_lambda=l2_lambda
        )

    print("W Matrisi (L2 Cezası Dahil) için Grad Check:")
    grad_check_sparse(f_weights, W, W.grad, num_checks=5)