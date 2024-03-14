import jittor as jt
from jittor import nn

class upfirdn2d(nn.Module):
    def __init__(self, kernel, up=1, down=1, pad=(0, 0)):
        super().__init__()

        self.kernel=kernel
        self.up=up
        self.down=down
        self.pad=pad

    def execute(self, input):
        up=self.up
        down=self.down
        pad=self.pad
        
        out = upfirdn2d_native(
            input, self.kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )
#        print("upfirdn2d", jt.flatten(out)[0])
        return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
#    print("upfirdn2d args", up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
#    print("upfirdn2d-1", jt.flatten(out)[0])
    out = nn.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
#    print("upfirdn2d-2", jt.flatten(out)[0])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
#    print("upfirdn2d-3", jt.flatten(out)[0])
    out = nn.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
#    print("upfirdn2d-4", jt.flatten(out)[0])
    out = out[
        :,
        max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
        :,
    ]
#    print("upfirdn2d-5", jt.flatten(out)[0])
    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = jt.misc.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
#    print(kernel)
#    print(w)
#    print("upfirdn2d-6", jt.flatten(w)[0])
    out = nn.conv2d(out, w)
#    print("upfirdn2d-7", jt.flatten(out)[0])
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
#    print("upfirdn2d-8", out_h)
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
#    print("upfirdn2d-9", out_w)
    return out.view(-1, channel, out_h, out_w)
