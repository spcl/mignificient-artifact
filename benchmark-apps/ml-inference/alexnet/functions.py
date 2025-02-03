import torch
import os
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer

import sys

model = None
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, os.path.pardir, os.path.pardir, os.path.pardir, 'benchmark-inputs', 'ml-inference', 'alexnet')

def function(obj):

    global model

    if model is None:

        #print("Model load", flush=True)
        #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model = torch.load(os.path.join(INPUT_DIR, 'alexnet.pt'))
        #before = timer()
        model.eval()
        model.to('cuda')
        #after = timer()

        #print('model eval time:', flush=True)
        #print(after - before)

    #start = timer()

    input_image = Image.open(os.path.join(INPUT_DIR, 'dog.jpg'))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    input_batch = input_batch.to('cuda')


    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    top_catid = top_catid[0].item()
    top_prob = top_prob[0].item()
    print("RESULT", top_prob, top_catid, flush=True)

    #end = timer()
    #print(end - start)

if __name__ == "__main__":

    for i in range(11):
        start = timer()
        function({})
        end = timer()
        print("Time:", end-start)
