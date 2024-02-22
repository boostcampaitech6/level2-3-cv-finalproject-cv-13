import numpy as np
import os,sys
import torch

from model import create_model
import yaml

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


def data_processing(plane):
    
    sample_path = os.path.join('data/valid', plane, '1130.npy')
    img_array = np.load(sample_path)

    img_array = np.stack((img_array,)*3, axis=1)
    img_array = torch.FloatTensor(img_array)
    img_array = img_array.unsqueeze(0)
    
    return img_array


def load_model(model_path, model_name, model_params, device): 

    model = create_model(model_name, **model_params)

    if os.path.exists(model_path): 
        model = torch.load(model_path, map_location=device)
    else:
        print("해당 경로에 모델 파일이 없습니다.")   

    return model


def run(config):
    PLANE = config['PLANE']
    MODEL_ROOT = config['MODEL_ROOT']
    MODEL = config['MODEL']
    model_name = MODEL['name']
    model_params = MODEL['params'] or {}

    if not os.path.exists('gradcamimages'):
        os.makedirs('gradcamimages')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(MODEL_ROOT, model_name, model_params, device)
    model.eval()

    image = data_processing(PLANE)
    target_layers = [model.target]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [BinaryClassifierOutputTarget(1)]

        cam_result_list = cam(
                            input_tensor=image.float(), targets=targets, 
                            aug_smooth=True, eigen_smooth=True
                                )
        original_image_list = torch.squeeze(image, dim=0).permute(0, 2, 3, 1).cpu().numpy() / 255.0

        visualization_list = [show_cam_on_image(original_image, cam_result) for original_image, cam_result in zip(original_image_list, cam_result_list)]
        visualization_images = [Image.fromarray(visualization) for visualization in visualization_list]

        for i, vis_res in enumerate(visualization_images):
            vis_res.save(f'gradcamimages/{i}.png')


if __name__ == '__main__':
    for config_file in os.listdir('./configs'):
        with open(f'./configs/{config_file}', 'r') as file:
            config = yaml.safe_load(file)
        run(config)