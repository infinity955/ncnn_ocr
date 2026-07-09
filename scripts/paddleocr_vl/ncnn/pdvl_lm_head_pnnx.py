# pnnx model stat
# model inputshape = [1,1,1024]f32
# FLOPS = 211.812M
# memory OPS = 106.011M

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = nn.Linear(bias=False, in_features=1024, out_features=103424)

        archive = zipfile.ZipFile('D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/paddleocr_vl/ncnn/pdvl_lm_head.pnnx.bin', 'r')
        self.linear.weight = self.load_pnnx_bin_as_parameter(archive, 'linear.weight', (103424,1024), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.linear(v_0)
        return v_1

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 1024, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/paddleocr_vl/ncnn/pdvl_lm_head_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 1024, dtype=torch.float)

    torch.onnx.export(net, v_0, "D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/paddleocr_vl/ncnn/pdvl_lm_head_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 1024, dtype=torch.float)

    import pnnx
    pnnx.export(net, "D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/paddleocr_vl/ncnn/pdvl_lm_head_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 1024, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
