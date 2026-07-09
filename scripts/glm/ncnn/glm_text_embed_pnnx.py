# pnnx model stat
# model inputshape = [1,?]i64
# FLOPS = 0
# memory OPS = 0

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

        self.embed = nn.Embedding(embedding_dim=1536, num_embeddings=59392, sparse=False)

        archive = zipfile.ZipFile('D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/glm/ncnn/glm_text_embed.pnnx.bin', 'r')
        self.embed.weight = self.load_pnnx_bin_as_parameter(archive, 'embed.weight', (59392,1536), 'float32')
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
        v_1 = self.embed(v_0)
        return v_1

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, -1), dtype=torch.long)

    mod = torch.jit.trace(net, v_0)
    mod.save("D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/glm/ncnn/glm_text_embed_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, -1), dtype=torch.long)

    torch.onnx.export(net, v_0, "D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/glm/ncnn/glm_text_embed_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, -1), dtype=torch.long)

    import pnnx
    pnnx.export(net, "D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/glm/ncnn/glm_text_embed_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.randint(10, (1, 16), dtype=torch.long)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
