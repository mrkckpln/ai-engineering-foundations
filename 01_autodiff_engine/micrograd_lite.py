import math
import random
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Value(-self.data, (self,), "neg")
        def _backward():
            self.grad += -1.0 * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    def relu(self):
        out = Value(max(0, self.data), (self,), "reLU")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
class Neuron:
    def __init__(self, nin, nonlin=True):
        # Küçük rastgele ağırlıklar (He initialization ölçeği)
        self.w = [Value(math.sqrt(2.0 / nin) * (2 * random.random() - 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act


class Layer:
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # Son katmanda nonlin=False yaparak çıktıyı serbest bırakıyoruz
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

if __name__ == "__main__":
    # 4 Örnekli 3 Boyutlu Mini Veri Seti
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # Hedef sınıflandırma değerleri

    # Model: 3 Girdi -> 4 Nöron -> 4 Nöron -> 1 Çıktı
    model = MLP(3, [4, 4, 1])

    # Eğitim Döngüsü (Gradient Descent)
    learning_rate = 0.05
    for epoch in range(50):
        # 1. Forward Pass
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        # 2. Zero Grad (Gradyan Sıfırlama)
        for p in model.parameters():
            p.grad = 0.0
            
        # 3. Backward Pass
        loss.backward()
        
        # 4. Parametre Güncelleme (SGD Step)
        for p in model.parameters():
            p.data -= learning_rate * p.grad
            
        if epoch % 10 == 0 or epoch == 49:
            print(f"Epoch {epoch:2d} | Kayıp (Loss): {loss.data:.4f}")
            
    print("\nSon Tahminler:")
    for x, y_target in zip(xs, ys):
        print(f"Girdi: {x} -> Hedef: {y_target:+.1f} | Model Tahmini: {model(x).data:+.4f}")