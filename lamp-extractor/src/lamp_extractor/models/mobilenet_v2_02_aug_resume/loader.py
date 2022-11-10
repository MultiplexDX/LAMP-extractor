import yaml
import torch
import albumentations as A
from pkg_resources import resource_filename
import cv2
import importlib.util
from loguru import logger
from lamp_extractor.models.mobilenet_v2_02_aug_resume import model as model_module

def predict(model, transform, params, img):
    model.eval()
    with torch.no_grad():
        if params['input_channels'] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tranformation = transform(image=img)
        x_tensor = tranformation['image']
        batch = x_tensor.unsqueeze(0)
        output = model(batch)
        output = output.squeeze()
        
        h, w = img.shape[:2]
        output[:, 0] *= w
        output[:, 1] *= h
        return output

def load_model(config_path, model_weigths_path, transform_path, model_src_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config['device'] != 'cpu':
        device = torch.device('cpu')
        map_location=device
    else:
        map_location=None
    
    # Load model source
    #spec = importlib.util.spec_from_file_location("model", model_src_path)
    #model_module = importlib.util.module_from_spec(spec)
    
    #spec.loader.exec_module(model_module)
    model = model_module.Net()
    checkpoint = torch.load(model_weigths_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("EPOCH", checkpoint['epoch'], "TRAIN LOSS", checkpoint['loss_train'], "VALID LOSS", checkpoint['loss_valid'])

    transform = A.load(transform_path, data_format="yaml")
    return model, config, transform

def _load_pyfunc():
    from pathlib import Path
    model_folder = resource_filename("lamp_extractor", "models/mobilenet_v2_02_aug_resume")
    model_folder = Path(model_folder)
    MODEL_WEIGTHS_PATH = model_folder / "best_model_weights.pth"
    TRANSFORM_PATH = model_folder / "infer_transform.yaml"
    CONFIG_PATH = model_folder / "config.yaml"
    MODEL_SRC_PATH = model_folder / "model.py"
    return load_model(
        config_path=CONFIG_PATH, 
        model_weigths_path=MODEL_WEIGTHS_PATH, 
        transform_path=TRANSFORM_PATH,
        model_src_path=MODEL_SRC_PATH,
    )

if __name__ == "__main__":
    from pkg_resources import resource_filename
    import cv2
    import importlib.util
    img_path = "F:/aston/Aston ITM spol. s r.o/MultiplexDX - LAMP TESTS - Dokumenty/Development/10.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    model_folder = resource_filename("lamp_extractor", "models/mobilenet_v2")
    model, config, transform = _load_pyfunc(model_folder)
    
    print(predict(model, transform, config, img))
