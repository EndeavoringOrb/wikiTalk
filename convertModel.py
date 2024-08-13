import torch
import struct
from model import *


def serializeMatrices(data, filename):
    with open(filename, "wb") as f:
        # Write the number of matrices
        f.write(struct.pack("I", len(data)))

        # Write each matrix
        for matrix in data:
            rows = matrix.shape[0]
            if len(matrix.shape) > 1:
                cols = matrix.shape[1]
            else:
                cols = 1
            # Write the rows and cols
            f.write(struct.pack("I", rows))
            f.write(struct.pack("I", cols))

            # Write the data
            f.write(matrix.cpu().detach().numpy().tobytes())


if __name__ == "__main__":
    modelPath = "models/embedArticle/0"

    model: RNNEmbedder = torch.load(
        f"{modelPath}/model.pt", map_location="cpu", weights_only=False
    )

    # save titleModel
    serializeMatrices([model.titleModel.embedding, model.titleModel.ih, model.titleModel.hh, model.titleModel.bias], f"{modelPath}/titleModel.bin")

    # save textModel
    serializeMatrices([model.textModel.embedding, model.textModel.ih, model.textModel.hh, model.textModel.bias], f"{modelPath}/textModel.bin")