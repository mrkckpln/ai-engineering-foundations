import math
import pytest
import torch
from micrograd_lite import Value


def test_finite_differences_gradient_check():
    """Limit tanımıyla sayısal gradyan (finite difference) hesabı ve

    analitik gradyanın (backward) karşılaştırılması.
    Formül: f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
    """
    h = 1e-5

    # Analitik gradyan hesabı (Bizim motor)
    a = Value(1.5)
    b = Value(-2.0)
    # f(a, b) = (a * b) + (a ** 2) + relu(b + 3)
    out = (a * b) + (a ** 2) + (b + 3.0).relu()
    out.backward()

    analytical_grad_a = a.grad
    analytical_grad_b = b.grad

    # Sayısal gradyan hesabı (a için)
    f_a_plus = ((1.5 + h) * -2.0) + ((1.5 + h) ** 2) + max(0.0, -2.0 + 3.0)
    f_a_minus = ((1.5 - h) * -2.0) + ((1.5 - h) ** 2) + max(0.0, -2.0 + 3.0)
    numerical_grad_a = (f_a_plus - f_a_minus) / (2 * h)

    # Sayısal gradyan hesabı (b için)
    f_b_plus = (1.5 * (-2.0 + h)) + (1.5 ** 2) + max(0.0, (-2.0 + h) + 3.0)
    f_b_minus = (1.5 * (-2.0 - h)) + (1.5 ** 2) + max(0.0, (-2.0 - h) + 3.0)
    numerical_grad_b = (f_b_plus - f_b_minus) / (2 * h)

    # Tolerans kontrolü (1e-5 hassasiyet)
    assert abs(analytical_grad_a - numerical_grad_a) < 1e-4, (
        f"a gradyan hatası: Analitik={analytical_grad_a}, Sayısal={numerical_grad_a}"
    )
    assert abs(analytical_grad_b - numerical_grad_b) < 1e-4, (
        f"b gradyan hatası: Analitik={analytical_grad_b}, Sayısal={numerical_grad_b}"
    )


def test_pytorch_parity_complex_dag():
    """Karmaşık bir DAG üzerinde operatörlerin (+, -, *, /, **, relu)

    PyTorch autograd sonuçlarıyla virgülden sonra birebir karşılaştırılması.
    """
    # 1. Bizim Motorumuzla Hesaplama
    a = Value(-3.0)
    b = Value(2.0)
    c = a + b                     # -1.0
    d = a * b + b ** 3            # -6.0 + 8.0 = 2.0
    c = c + c + 1                 # -1.0 + -1.0 + 1.0 = -1.0
    c = c + 1 + c + (-a)          # -1.0 + 1.0 + -1.0 + 3.0 = 2.0
    d = d + d * 2 + (b + a).relu()  # 2.0 + 4.0 + relu(-1.0) = 6.0
    d = d + 3 * d + (b - a).relu()  # 6.0 + 18.0 + relu(5.0) = 29.0
    e = c - d                     # 2.0 - 29.0 = -27.0
    f = e ** 2                    # 729.0
    g = f / 2.0                   # 364.5
    g = g + 10.0 / f              # 364.5 + (10 / 729)
    g.backward()

    # 2. PyTorch ile Hesaplama (Double Precision 64-bit float)
    pa = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    pb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    pc = pa + pb
    pd = pa * pb + pb ** 3
    pc = pc + pc + 1
    pc = pc + 1 + pc + (-pa)
    pd = pd + pd * 2 + torch.relu(pb + pa)
    pd = pd + 3 * pd + torch.relu(pb - pa)
    pe = pc - pd
    pf = pe ** 2
    pg = pf / 2.0
    pg = pg + 10.0 / pf
    pg.backward()

    # İleri akış (Forward) kontrolü
    assert abs(g.data - pg.item()) < 1e-6, f"Forward uyuşmazlığı: {g.data} != {pg.item()}"

    # Geri akış (Backward / Gradient) kontrolleri
    assert abs(a.grad - pa.grad.item()) < 1e-6, f"a.grad uyuşmazlığı: {a.grad} != {pa.grad.item()}"
    assert abs(b.grad - pb.grad.item()) < 1e-6, f"b.grad uyuşmazlığı: {b.grad} != {pb.grad.item()}"


def test_gradient_accumulation():
    """Aynı değişkenin grafikte birden fazla dalda kullanılması durumunda

    gradyanların += ile doğru toplanıp toplanmadığının testi (x + x durumu).
    """
    x = Value(4.0)
    y = x + x  # dy/dx = 2 olmalı
    y.backward()
    assert x.grad == 2.0, f"Beklenen gradyan 2.0, bulunan: {x.grad}"

    a = Value(3.0)
    b = a * a  # db/da = 2a = 6.0 olmalı
    b.backward()
    assert a.grad == 6.0, f"Beklenen gradyan 6.0, bulunan: {a.grad}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])