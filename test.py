import numpy as np
import torch
import cv2
import random
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision import transforms

DEVICE = 'cuda'


def video_test():
    cap_div = cv2.VideoCapture("20190923_144136.mp4")
    model = TestKeypointRcnn(1000)
    while True:
        r, f = cap_div.read()
        # display_image(f, 10)
        model(f, 50)
    cv2.destroyAllWindows()


class TestKeypointRcnn():
    def __init__(self, size=480):
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.CenterCrop(size),
                                             transforms.Resize(512),
                                             transforms.ToTensor()])
        # torch_tensor_to_img(img, display=True)
        model = keypointrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        model.to(DEVICE)
        self.model = model

    def __call__(self, img=None, time=0):
        if img is None:
            img = cv2.imread("test.jpg")
        img = self.transform(img)
        predictions = self.model([img.to(DEVICE)])[0]
        predictions_count = predictions['boxes'].shape[0]
        img = torch_tensor_to_img(img)
        kp_colours = {}
        for idx in range(predictions_count):
            # print(predictions['boxes'][idx].cpu().detach().numpy())
            point = predictions['boxes'][idx].cpu().detach().numpy()
            if predictions['scores'][idx] > 0.5:
                colour = [int((255/predictions_count) * idx), 255, 255]
                cv2.rectangle(img, tuple(point[:2]), tuple(point[2:]), tuple(colour), 1)
                x = (point[2] + point[0]) / 2
                y = point[3]
                cv2.circle(img, (int(x), int(y)), 1, colour, 1, 1)
                for kp_idx, kp in enumerate(predictions['keypoints'][idx]):
                    continue
                    if kp_idx not in [14, 13]:
                        continue
                    kp = kp.cpu().detach().numpy()
                    try:
                        colour = kp_colours[kp_idx]
                    except KeyError:
                        colour = tuple(random.randint(0, 255) for _ in range(3))
                        kp_colours[kp_idx] = colour
                    cv2.circle(img, tuple(kp[:2].astype(np.int)), 1, colour, 1, 1)
        display_image(img, time)


def torch_tensor_to_img(image, display=False):
    image_convereted = (image.numpy() * 255).astype(np.uint8).transpose(1, 2, 0).copy()
    if display:
        display_image(image)
    return image_convereted


def display_image(image, time=0):
    cv2.imshow("", image)
    cv2.waitKey(time)
    if time == 0:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # TestKeypointRcnn()()
    video_test()
