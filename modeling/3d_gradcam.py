import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import code


def run(INPUT_DATA_PATH, MODEL_PATH, EXP_NAME, THRESHOLD, USE_CONTOUR):
    os.makedirs(f'./cam_results/{EXP_NAME}', exist_ok=True)
    os.makedirs(f'./cam_results/{EXP_NAME}/score', exist_ok=True)
    os.makedirs(f'./cam_results/{EXP_NAME}/images', exist_ok=True)
    
    model = torch.load(MODEL_PATH)
    target_layers = [model.model.blocks[4].res_blocks[0].branch2.conv_c]
    # target_layers = model.target
    # target_layers = [model.pretrained_model.features[-1]]
    
    array = np.load(INPUT_DATA_PATH)
    images = np.stack((array,)*3, axis=1)
    array = np.stack((array,)*3, axis=0)
    array = np.expand_dims(array, axis=0)

    input_tensor  = torch.FloatTensor(array)

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(1)]

    cam_results = cam(input_tensor=input_tensor, targets=targets, cam_score=False)
    cam_score_results = cam(input_tensor=input_tensor, targets=targets, cam_score=True)

    cam_results = np.squeeze(cam_results)
    cam_score_results = np.squeeze(cam_score_results)
    # code.interact(local=dict(globals(), **locals()))

    cam_scores = []

    writer = open(f'./cam_results/{EXP_NAME}/score/cam_score.csv', 'w')
    print('idx,cam_score', file=writer)
    for i, (image, cam_result, cam_score_result) in enumerate(zip(images, cam_results, cam_score_results)):
        # code.interact(local=dict(globals(), **locals()))
        cam_score = cam_score_result.max()
        cam_scores.append(cam_score)
        print(f'{i},{cam_score}', file=writer)
        image = image / 255.0
        image = image.transpose(1, 2, 0)
        visualization = show_cam_on_image(image, cam_result, use_rgb=True, threshold=THRESHOLD, use_contour=USE_CONTOUR)
        image = Image.fromarray(visualization)
        image.save(f'./cam_results/{EXP_NAME}/images/{i}.png')

    plt.figure(figsize=(12, 8))

    plt.plot(cam_scores, linewidth=5, color='black')
    max_score_index = cam_scores.index(max(cam_scores))
    plt.axvline(x=max_score_index, color='red', linestyle='--')
    plt.plot(max_score_index, cam_scores[max_score_index], 'ro')

    plt.xticks([i for i in range(len(cam_scores))])
    plt.savefig(f'./cam_results/{EXP_NAME}/score/cam_score.png')
    plt.close()


if __name__ == "__main__":
    with open('cam_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    INPUT_DATA_PATH = config['INPUT_DATA_PATH']
    MODEL_PATH = config['MODEL_PATH']

    THRESHOLD = config['THRESHOLD']
    USE_CONTOUR = config['USE_CONTOUR']

    EXP_NAME = config['EXP_NAME']

    run(INPUT_DATA_PATH, MODEL_PATH, EXP_NAME, THRESHOLD, USE_CONTOUR)
