import logging
import torch

if __name__ == '__main__':
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load('some-model.pt', map_location=device)['model']
    model.float().eval()    # inference mode

    # use float16 tensors if cuda is available
    # this sets us up for real-time inference with GPUs later
    if torch.cuda.is_available():
        model.half().to(device)

