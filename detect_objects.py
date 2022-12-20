import torch.cuda

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from torchvision import transforms


def detect_objects(device, model, image):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)

    # use float16 tensors if cuda is available
    # this sets us up for real-time inference with GPUs later
    if torch.cuda.is_available():
        image = image.half().to(device)

    image = image.unsqueeze(0)
    with torch.no_grad():
        output, _ = model(image)

    output = non_max_suppression_kpt(output,
                                     0.25,
                                     0.65,
                                     nc=model.yaml['nc'],
                                     nkpt=model.yaml['nkpt'],
                                     kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

    return output

# https://stackabuse.com/real-time-pose-estimation-from-video-in-python-with-yolov7/

