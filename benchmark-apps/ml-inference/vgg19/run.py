import torch
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
before = timer()
model.eval()
model.to('cuda')
after = timer()

print('model eval time:')
print(after - before)

start = timer()
input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

input_batch = input_batch.to('cuda')

with torch.no_grad():
    output = model(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
top_prob, top_catid = torch.topk(probabilities, 1)
top_catid = top_catid[0].item()
top_prob = top_prob[0].item()

print(top_prob, top_catid)

end = timer()
print(end - start)

import signal, time

def signal_handler(signum, frame):
    print("\nSignal received. Exiting gracefully.")
    exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

print("Running... Press Ctrl+C to exit.")

try:
    # This will run indefinitely until a SIGINT is received
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # This block will not be executed due to our signal handler
    pass

print("This line will not be reached.")
