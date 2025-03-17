import torch
import os
from model.u2net import U2NET, U2NETP
from torch import nn
from torchvision import transforms
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import SalObjDataset
from torch.utils.data import  DataLoader
import glob
from torch.autograd import Variable
from PIL import Image, ImageDraw
import numpy as np
import argparse
import configparser



bce_loss = nn.BCELoss(reduction='mean')

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def print_prediction_metric(pred):
    mean_pred = torch.mean(pred).item()
    return mean_pred

def load_image(image_path):
    img_name_list = glob.glob(image_path + os.sep + '*')
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensor()]),
                                        mode='test'
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    return test_salobj_dataloader, img_name_list


def save_combined_output(image_name, predictions, model_names, original_labels, output_path):
    def process_prediction(pred):
        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy()
        if np.max(predict_np) > 1:
            predict_np = predict_np / 255.0
        return Image.fromarray((predict_np * 255).astype(np.uint8))

    def create_composite(image, mask):
        mask = mask.convert('L')
        return Image.composite(image, Image.new('RGB', image.size, color=(255, 0, 0)), mask)

    original_image = Image.open(image_name).convert('RGB')
    total_width = original_image.width * (len(predictions) + 1)
    combined_im = Image.new('RGB', (total_width, original_image.height))

    combined_im.paste(original_image, (0, 0))
    draw = ImageDraw.Draw(combined_im)
    draw.text((10, 10), "Original", fill='black')

    for idx, (pred, model_name) in enumerate(zip(predictions, model_names)):
        processed_matte = process_prediction(pred).resize(original_image.size, resample=Image.BILINEAR)
        composite = create_composite(original_image, processed_matte)
        combined_im.paste(composite, (original_image.width * (idx + 1), 0))

        loss = normPRED(pred)
        mean_pred = print_prediction_metric(loss)
        draw.text((original_image.width * (idx + 1) + 10, 10), f"{model_name} Loss: {mean_pred:.4f}", fill='green')

    img_name = os.path.basename(image_name).split('.')[0]
    save_path = os.path.join(output_path, f"{img_name}_combined.png")
    combined_im.save(save_path)



def process_image(net, image):
    inputs_test = image.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
    d0 = d1 + d2 + d3 + d4 + d7
    pred = d0[:, 0, :, :]
    pred = normPRED(pred)
    mean_pred = print_prediction_metric(pred)
    return pred, mean_pred



def load_model(model_type, model_weight, gpu):
    torch.cuda.set_device(gpu)
    print("Current device:", torch.cuda.current_device())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type.lower() == 'u2net':
        net = U2NET(3, 1)
    elif model_type.lower() == 'u2netp':
        net = U2NETP(3, 1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if torch.cuda.is_available():
        net.cuda(device=device)
        state_dict = torch.load(model_weight)
        reversed_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(reversed_state_dict)
        net.eval()
    # else:
    #     net.load_state_dict(torch.load(model_weight, map_location='cpu'))
    #     net.eval()

    return net

def argument_parser():
    parser = argparse.ArgumentParser(description='Model testing')
    parser.add_argument('-l', '--list', nargs='+', help="list of weights", required=True)
    return parser.parse_args()


def load_configuration():
    config_file = os.getenv('CONFIG_FILE', 'config.ini')

    config = configparser.ConfigParser()
    config.read(config_file)

    print(f"Using config file: {config_file}")

    input_path = config.get('testing', 'input')
    output_path = config.get('testing', 'output')
    model_type = config.get('model_settings', 'model_type')
    device = config.getint('cuda', 'gpu')
    
    return {
        "model_type": model_type,
        "cuda": device,
        "input": input_path,
        "output": output_path
    }
    
def main():
    args = argument_parser()
    config = load_configuration()
    test_loader, img_name_list = load_image(config["input"])

    for j, data_test in enumerate(test_loader):
        image = data_test['image']
        original_label = data_test.get('label')
        all_model_predictions = []
        all_model_names = []

        for model_weight in args.list:
            model_name = os.path.basename(model_weight).split('.')[0]
            net = load_model(config['model_type'], model_weight, config['cuda'])

            pred, mean = process_image(net, image)
            all_model_predictions.append(pred)
            all_model_names.append(model_name)

        save_combined_output(img_name_list[j], all_model_predictions, all_model_names, original_label, config['output'])

if __name__ == '__main__':
    main()

