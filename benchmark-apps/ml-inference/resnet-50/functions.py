import torch
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer

import sys
print(sys.path)


print(torch.__path__, flush=True)

def function(obj):

    print("Model load", flush=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
# model = torch.load('resnext50_32x4d.pt')

    print("Model done", flush=True)
    before = timer()
    model.eval()
    print("Model finished", flush=True)
    model.to('cuda')
    print("Model finished to CUDA", flush=True)
    torch.cuda.get_device_properties('cuda')
    after = timer()

    print('model eval time:', flush=True)
    print(after - before)

    start = timer()

    print("Read", flush=True)
    input_image = Image.open('/work/serverless/2024/gpus/mignificient-artifact/benchmark-apps/ml-inference/resnet-50/dog.jpg')
    print("Read2", flush=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("preprocess", flush=True)
    input_tensor = preprocess(input_image)
    print("input_tensor", flush=True)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    print("unsqueeze", flush=True)

    print("to cuda call", flush=True)
    input_batch = input_batch.to('cuda')
    print("to cuda done", flush=True)


    with torch.no_grad():
        output = model(input_batch)
        print("model done", flush=True)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print("result done", flush=True)
    top_prob, top_catid = torch.topk(probabilities, 1)
    print("result done2", flush=True)
    print(top_prob, top_catid, flush=True)
    print("result done2.5", flush=True)
    top_catid = top_catid[0].item()
    print("result done3", flush=True)
    top_prob = top_prob[0].item()
    print("result done4", top_prob, top_catid, flush=True)

    end = timer()
    print(end - start)
