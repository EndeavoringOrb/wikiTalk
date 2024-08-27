import os
import json
import numpy as np
import math
import torch
import pickle
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from helperFuncs import *
from tqdm import tqdm, trange
from model import *
from vocab import *
from tokenizeWiki import loadTokens, loadTitles, countNumTokens
from time import perf_counter
from optimizer import CustomAdam

from line_profiler import profile
import os

os.environ["LINE_PROFILE"] = "0"


def getNumPages(folder):
    with open(f"{folder}/info.txt", "r", encoding="utf-8") as f:
        text = f.read()
        numPages = int(text.split(" ")[0].strip())
    return numPages


def getEmbeddingInit(rows, cols, numSteps):
    data = torch.normal(0, 0.02, (rows, cols))
    vectors = torch.nn.Parameter(data)
    optimizer = torch.optim.Adam([vectors], lr=0.001)
    mask = torch.zeros(rows, rows).fill_diagonal_((float("inf")))

    for _ in trange(numSteps):
        optimizer.zero_grad()

        # normalize vectors
        normVal = (1 / vectors.norm(dim=1)).unsqueeze(-1)
        normVecs = vectors * normVal

        # Compute pairwise distances
        distances = torch.cdist(normVecs, normVecs)

        # Set diagonal to a large value to exclude self-distances
        distances = distances + mask

        # Find the minimum distance for each vector
        min_distances = distances.min(dim=1).values

        # Compute the total distance (negative sum of minimum distances)
        totalDist = -min_distances.sum()

        totalDist.backward()
        optimizer.step()

    print(f"{totalDist.item() / (2 * rows)}")

    vectors.data /= vectors.norm(dim=1).unsqueeze(-1)

    return vectors


@profile
def compute_jacobian_params2(output, input, model_zeros, device):
    grad_output = torch.eye(output.size(1), device=device).unsqueeze(dim=1)
    jacobian = torch.autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=grad_output,
        retain_graph=True,
        create_graph=True,
        is_grads_batched=True,
        allow_unused=True,
    )
    jacs = []
    for j, jac in enumerate(jacobian):
        if jac is not None:
            jacs.append(jac.view(output.size(1), -1))
        else:
            jacs.append(model_zeros[j].repeat(output.size(1), 1))
    return torch.cat(jacs, dim=-1).detach()


@profile
def compute_jacobian2(output, input, device):
    grad_output = torch.eye(output.size(1), device=device).unsqueeze(dim=1)
    jacobian = torch.autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=grad_output,
        retain_graph=True,
        create_graph=True,
        is_grads_batched=True,
        allow_unused=True,
    )
    return jacobian[0]


def backPropHidden():
    # through ln_f
    pass


@profile
def main():
    # Hyperparameters
    vocabSize = len(vocab)
    hiddenSize = 4
    numEpochs = 1_000_000
    learningRate = 2e-4
    batchSize = 1
    nHead = 2
    headSize = 2
    nLayer = 1

    # Settings
    modelLoadPath = "models/tokenPredArticle2/1_0"
    modelSavePath = "models/tokenPredArticle2/current"
    saveInterval = 1
    tokenFolder = "tokenData"

    device = torch.device(
        "cpu"
    )  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training Parameters:")
    print(f"# of Epochs: {numEpochs:,}")
    print(f"Learning Rate: {learningRate:,}")
    print(f"Batch Size: {batchSize:,}")
    print(f"Training Device: {device}")
    print()

    # Initialize the model, loss function, and optimizer
    print("Initializing model...")
    # model: RNNLanguage = torch.load(
    #    f"{modelLoadPath}/model.pt", weights_only=False, map_location=device
    # )
    model: RecurrentTransformer = RecurrentTransformer(
        vocabSize, hiddenSize, nHead, headSize, nLayer, device
    )
    nParams = sum([p.numel() for p in model.parameters()])

    # create optimizer
    optimizer = CustomAdam(nParams, device, learningRate)

    clearLines(1)
    print(f"Model Parameter Information:")
    print(f"Vocab Size: {model.vocabSize:,}")
    print(f"Hidden Dim: {model.hiddenSize:,}")
    nParams = sum([p.numel() for p in model.parameters()])
    print(f"Model Total # Params: {nParams:,}")
    print()

    # initialize things for training
    params = list(model.parameters())
    model_zeros = [
        torch.zeros(
            torch.prod(torch.tensor([*param.shape])).unsqueeze(0), device=param.device
        )
        for param in params
    ]
    dL_dP = torch.zeros(nParams)
    delta = torch.zeros(hiddenSize, nParams)
    dR_dPCurrent = torch.zeros(hiddenSize, nParams)
    dR_dRPrev = torch.zeros(hiddenSize, hiddenSize)
    dL_dR = torch.zeros(hiddenSize)

    print(f"Model Weights:")
    lm_head_weight_index = -1
    lm_head_bias_index = -1
    paramIndex = 0

    for i, thing in enumerate(model.named_parameters()):
        if thing[0] == "lm_head.weight":
            lm_head_weight_index = paramIndex
        elif thing[0] == "lm_head.bias":
            lm_head_bias_index = paramIndex
        paramIndex += np.prod(thing[1].shape)
        weightShape = list(thing[1].shape)
        weightShapeString = ", ".join([str(num) for num in weightShape])
        print(f"{i}: {thing[0]}, ({weightShapeString})")
    print()

    # Get all titles
    print(f"Loading all page titles...")
    titles = []
    for _, _, titleTokens in loadTitles(tokenFolder):
        titles.append(titleTokens)
    print(f"Finished loading")
    print(f"Shuffling titles...")
    random.shuffle(titles)
    clearLines(3)
    print(f"Loaded {len(titles):,} titles")

    # Get number of pages
    print(f"Counting # of pages and # of tokens per epoch...")
    numPagesPerEpoch, numTokensPerEpoch = countNumTokens(tokenFolder)
    clearLines(1)
    print(f"{numPagesPerEpoch:,} pages per epoch")
    print(f"{numTokensPerEpoch:,} tokens per epoch")

    # Save model info
    print(f"Saving model info...")
    os.makedirs(f"{modelSavePath}", exist_ok=True)
    with open(f"{modelSavePath}/info.txt", "w", encoding="utf-8") as f:
        infoText = f"Vocab Size: {vocabSize}\n"
        infoText += f"Hidden Dim: {hiddenSize}\n"
        infoText += f"Learning Rate: {learningRate}\n"
        infoText += f"# Pages Per Epoch: {numPagesPerEpoch}\n"
        infoText += f"# Tokens Per Epoch: {numTokensPerEpoch}\n"
        f.write(infoText)

    with open(f"{modelSavePath}/loss.txt", "w", encoding="utf-8") as f:
        f.write("# Steps Trained, # Tokens Trained On, Loss\n")

    clearLines(1)
    print("Training...\n")

    totalStepNum = 0
    totalNumTokens = 0

    # Training loop
    for epoch in range(numEpochs):
        totalLoss = 0
        windowLoss = 0
        windowSteps = 0
        lastLoss = "N/A"
        lastTokSec = 0
        numPages = 0
        numTokens = 0

        tokenLoader = loadTokens(tokenFolder)

        numBatches = int(math.ceil(numPagesPerEpoch / batchSize))

        stepNum = 0

        for _ in range(numBatches):
            # Print progress, loss and tok/sec
            print(
                f"Epoch [{epoch+1}/{numEpochs}], Batch [{stepNum + 1}/{numBatches}] ({100.0 * (stepNum + 1) /numBatches:.4f}%), Last Loss: {lastLoss}, Last Tok/Sec: {lastTokSec}"
            )

            start = perf_counter()
            batch = []
            adjustedBatchSize = min(batchSize, numPagesPerEpoch - numPages)
            for i in range(adjustedBatchSize):
                fileIndex, titleTokens, textTokens = next(tokenLoader)
                batch.append((titleTokens, textTokens))
            numPages += adjustedBatchSize

            # sort batch by article length, and get lengths
            batch = sorted(batch, key=lambda x: len(x[1]))
            lengths = [len(item[1]) for item in batch]
            lengths.insert(0, 0)

            # Prepare for forward pass
            model.preCompute()  # Pre-Compute variables for a faster forward pass

            # Reset the states
            states = model.initState.expand(adjustedBatchSize, -1).requires_grad_()
            newStates = states.clone().requires_grad_()
            mask = torch.zeros(vocabSize)

            loss = 0
            batchStepNum = 0
            delta.fill_(0)
            dL_dP.fill_(0)

            # Batch process articles
            with tqdm(total=lengths[-1], desc="Forward Pass") as pbar:
                for i in range(len(lengths) - 1):
                    # init states and tokens for this length
                    tokens = []
                    for item in batch[i:]:
                        tokens.append(item[1][lengths[i] : lengths[i + 1]])
                    tokens = torch.tensor(tokens, device=device, dtype=torch.int64)

                    # train
                    for tokIdx in range(len(tokens[0])):
                        token = tokens[:, tokIdx]
                        if newStates.grad_fn != None:
                            with torch.no_grad():
                                # Get dR_dPCurrent
                                dR_dPCurrent = compute_jacobian_params2(
                                    newStates, params, model_zeros, device
                                )

                                # Get dR_dRPrev
                                dR_dRPrev = (
                                    compute_jacobian2(newStates, states, device)
                                    .transpose(0, 1)
                                    .sum(0)
                                )

                                # Update delta
                                delta = dR_dPCurrent + dR_dRPrev @ delta

                        # Get pred
                        pred = model.getPreds(newStates)

                        with torch.no_grad():
                            # Get dY_dR
                            dY_dR = (
                                compute_jacobian2(pred, newStates, device)
                                .transpose(0, 1)
                                .sum(0)
                            )

                            # Get probs
                            probs = F.softmax(pred, dim=-1)

                            # Update loss
                            currentLossVal = -torch.log(probs[:, token]).item()
                            loss += currentLossVal

                            # Get dL_dP
                            dL_dY = probs.clone().sum(0)
                            mask[token] = 1
                            dL_dY = dL_dY - mask

                            # Update dY_dP
                            dY_dP = dY_dR @ delta
                            for vocIdx in range(vocabSize):
                                dY_dP[
                                    vocIdx,
                                    lm_head_bias_index : lm_head_bias_index + vocIdx,
                                ] += 1
                                dY_dP[
                                    vocIdx,
                                    lm_head_weight_index
                                    + vocIdx * hiddenSize : lm_head_weight_index
                                    + (vocIdx + 1) * hiddenSize,
                                ] += newStates.sum(0)

                            # Update dL_dP
                            thing0 = dL_dP.count_nonzero() / dL_dP.numel()
                            thing1 = dL_dY.count_nonzero() / dL_dY.numel()
                            thing2 = dY_dP.count_nonzero() / dY_dP.numel()
                            dL_dP += dL_dY @ dY_dP

                        # Get next state
                        states = newStates.detach().requires_grad_()
                        newStates = model.nextState(states, token)

                        pbar.update(1)

                        mask[token] = 0

                        batchStepNum += 1

                    # update
                    # pbar.update(lengths[i + 1] - lengths[i])
                    with torch.no_grad():
                        states = states[1:]
                        newStates = newStates[1:]

            loss /= lengths[-1]

            # Manually update the model parameters using gradient descent
            with torch.no_grad():
                dL_dP *= 1.0 / lengths[-1]
                grads = optimizer.get_grads(dL_dP)

                start_idx = 0
                for param in params:
                    param_length = param.numel()
                    param_gradient = grads[
                        start_idx : start_idx + param_length
                    ].view_as(param)
                    param -= param_gradient
                    start_idx += param_length

                # Reset for next sequence
                dL_dP.fill_(0)

            # Track loss and counters
            stop = perf_counter()
            totalLoss += loss
            lastLoss = loss
            windowLoss += loss
            windowSteps += 1
            numPages += adjustedBatchSize

            batchNumTokens = sum([len(item[1]) for item in batch])
            lastTokSec = int(batchNumTokens / (stop - start))
            numTokens += batchNumTokens

            totalStepNum += 1
            totalNumTokens += batchNumTokens

            # Save the trained model
            if stepNum % saveInterval == 0 and stepNum != (numBatches - 1):
                print("Saving model...")
                torch.save(model, f"{modelSavePath}/model.pt")
                with open(f"{modelSavePath}/loss.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"{totalStepNum}, {totalNumTokens}, {windowLoss / windowSteps}\n"
                    )
                windowLoss = 0
                windowSteps = 0
                clearLines(1)

            # Clear logging so we are ready for the next step
            clearLines(2)

            stepNum += 1

            # FOR TESTING ONLY
            # Stop after first few batches to see if we can overfit
            # if stepNum == 1:
            #    break

        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {totalLoss/stepNum:.4f}")

        # Save the trained model
        torch.save(model, f"{modelSavePath}/model_E{epoch + 1}.pt")


if __name__ == "__main__":
    main()
