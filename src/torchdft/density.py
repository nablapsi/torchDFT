class Density:
    """Density data structure."""

    def __init__(self, density, grad=None):
        self.value = density
        self.grad = grad

    def detach(self):
        value = self.value.detach()

        if self.grad is not None:
            grad = self.grad.detach()
        else:
            grad = None
        return Density(value, grad)
