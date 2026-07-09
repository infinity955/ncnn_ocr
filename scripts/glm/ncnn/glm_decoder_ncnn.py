import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 16, 1536, dtype=torch.float)
    in1 = torch.rand(1, 1, 16, 16, dtype=torch.float)
    in2 = torch.rand(1, 16, 64, dtype=torch.float)
    in3 = torch.rand(1, 16, 64, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/glm/ncnn/glm_decoder.ncnn.param")
        net.load_model("D:/MySystem/share/SummerNcnn/pnnx/ncnn_ocr/scripts/glm/ncnn/glm_decoder.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(in1.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(in2.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(in3.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
