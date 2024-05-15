# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from torch import Tensor, nn


class ConvGLUModule(nn.Module):
    """
    Conv -> GLU -> Conv -> BatchNorm -> activation -> Conv
    """

    def __init__(self, hidden_size: int, kernel_size: int, activation: nn.Module):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0

        self.linear1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=hidden_size,
        )
        self.norm = nn.BatchNorm1d(hidden_size)  # FIXME: 危ないかも
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.activation = activation

    def forward(
        self,
        x: Tensor,  # (B, T, ?)
    ):
        x = nn.functional.glu(self.linear1(x), dim=2)  # (B, T, ?)
        x = x.transpose(1, 2)  # (B, ?, T)
        x = self.activation(self.norm(self.depthwise_conv(x)))  # (B, ?, T)
        x = x.transpose(1, 2)  # (B, T, ?)
        x = self.linear2(x)  # (B, T, ?)
        return x
