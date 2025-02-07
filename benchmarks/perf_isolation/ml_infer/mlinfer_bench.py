import multiprocessing
import re
import torch
import time
import concurrent.futures
import numpy as np
import argparse
import os


# append LD_LIBRARY_PATH with conda env
os.environ['LD_LIBRARY_PATH'] = f'/home/ctianche/miniconda3/envs/mig_dev/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

import subprocess
import signal
import atexit
from PIL import Image
from torchvision import transforms
from bert import QA
from yolodataset import LoadImages
from yologeneral import non_max_suppression



class GPUInstanceManager:
    def __init__(self, mode):
        self.mode = mode
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        self.cleanup()
        exit(0)
        
    def setup_mig(self, num_instances):
        print("Destroying existing MIG instances")
        # sudo nvidia-smi -i 0 -c DEFAULT
        # sudo nvidia-smi -i 0 -mig 1
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-c', 'DEFAULT'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-mig', '1'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dci'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dgi'], capture_output=True)
        
        gpu_profiles = {2: 9, 3: 14, 4: 15, 7: 19}
        profile = gpu_profiles.get(num_instances)
        if not profile:
            raise ValueError(f"Unsupported number of MIG instances: {num_instances}")
            
        # Create GPU instances
        for i in range(num_instances):
            result = subprocess.run(['sudo', 'nvidia-smi', 'mig', '-cgi', f'{profile}', '-C'], 
                                 capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create GPU instance: {result.stderr}")
            print(result.stdout)
        
    def setup_mps(self, num_instances):
        # Stop any existing MPS daemon
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-mig', '0'], capture_output=True)
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-c', 'EXCLUSIVE_PROCESS'], capture_output=True)
        
        # Start MPS daemon
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        instance_to_sm_ratio = {2: 3/7, 3: 2/7, 4: 1/7, 7: 1/7}
        if self.mode == 'mps_limit_sm':
            sm_ratio = int(instance_to_sm_ratio.get(num_instances) * 100)
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(sm_ratio)
            print(f"Setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to {sm_ratio}")
        subprocess.run(['nvidia-cuda-mps-control', '-d'])
        print("Started MPS")
        
    def cleanup(self):
        if self.mode == 'mig':
            subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dci'])
            subprocess.run(['sudo', 'nvidia-smi', 'mig', '-dgi'])
            print("Destroyed all MIG instances")
        else:
            # subprocess.run(['nvidia-cuda-mps-control', '-d'])
            subprocess.run('echo quit | nvidia-cuda-mps-control', shell=True)
            if self.mode == 'mps_limit_sm':
                if 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' in os.environ:
                    del os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']
            print("Stopped MPS")
            subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-c', 'DEFAULT'], capture_output=True)
            subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-mig', '1'], capture_output=True)

class BaseTest:
    def __init__(self, uu_id, **kwargs):
        self.uu_id = uu_id
        self.kwargs = kwargs
        
    def prepare(self, dev_str: str):
        raise NotImplementedError()
    
    def launch_gpu_work(self):
        raise NotImplementedError()
    
    def get_task_name(self):
        return "basetest"   
    
    def run_bench(self, num_iterations=32):
        # For MIG mode, set specific GPU
        if 'MIG' in str(self.uu_id):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.uu_id)
            
        device = 'cuda'
        self.prepare(device)
        
        # warmup
        for _ in range(4):
            self.launch_gpu_work()

        latencies = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            self.launch_gpu_work()
            torch.cuda.synchronize()
            latencies.append(time.time() - start_time)
            
        return {
            "latencies": latencies
        }

class AlexnetTest(BaseTest):
    def __init__(self, uu_id, **kwargs):
        super().__init__(uu_id, **kwargs)
    
    def get_task_name(self):
        return 'alexnet'
    
    def prepare(self, dev_str):
        self.model = torch.load("/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/alexnet/alexnet.pt", weights_only=False)
        self.model.eval()
        self.model.to(dev_str)

        input_image = Image.open('/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/alexnet/dog.jpg')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        self.input_batch = input_batch.to(dev_str)
    
    def launch_gpu_work(self):
        with torch.no_grad():
            output = self.model(self.input_batch)        
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
        return top_prob, top_catid

class Resnet50Test(BaseTest):
    def __init__(self, uu_id, **kwargs):
        super().__init__(uu_id, **kwargs)
    
    def get_task_name(self):
        return 'resnet50'
    
    def prepare(self, dev_str):
        self.model = torch.load("/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/resnet-50/resnet-50.pt", weights_only=False)
        self.model.eval()
        self.model.to(dev_str)

        input_image = Image.open('/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/resnet-50/dog.jpg')
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        self.input_batch = input_batch.to(dev_str)
    
    def launch_gpu_work(self):
        with torch.no_grad():
            output = self.model(self.input_batch)        
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
        return top_prob, top_catid
    
class BertSquadTest(BaseTest):
    def __init__(self, uu_id, **kwargs):
        super().__init__(uu_id, **kwargs)
    
    def get_task_name(self):
        return 'bertsquad'
    
    def prepare(self, dev_str):
        self.model = QA("/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/BERT-SQuAD/model")
    
    def launch_gpu_work(self):
        doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by " \
            "the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the " \
            "state's law-making body for matters coming under state responsibility. The Victorian Constitution can be " \
            "amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an " \
            "absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian " \
            "people in a referendum, depending on the provision. "
        q = 'When did Victoria enact its constitution?'

        answer = self.model.predict(doc, q)
        return answer['answer']

class Vgg19Test(BaseTest):
    def __init__(self, uu_id, **kwargs):
        super().__init__(uu_id, **kwargs)
    
    def get_task_name(self):
        return 'vgg19'
    
    def prepare(self, dev_str):
        self.model = torch.load("/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/vgg19/vgg19.pt", weights_only=False)
        self.model.eval()
        self.model.to(dev_str)

        input_image = Image.open('/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/vgg19/dog.jpg')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        self.input_batch = input_batch.to(dev_str)
    
    def launch_gpu_work(self):
        with torch.no_grad():
            output = self.model(self.input_batch)        
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
        return top_prob, top_catid

class YolopTest(BaseTest):
    def __init__(self, uu_id, **kwargs):
        super().__init__(uu_id, **kwargs)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.transform=transforms.Compose([
                    transforms.ToTensor(),
                    self.normalize,
                ])
            
    def get_task_name(self):
        return 'yolop'
    
    def prepare(self, dev_str):
        self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        self.model.eval()
        self.model.to(dev_str)

        self.dataset = LoadImages(
            '/home/ctianche/mignificient-artifact/benchmark-inputs/ml-inference/yolop/images',
            img_size=640
        )
        self.inputs = []
        for path, img, img_det, vid_cap,shapes in self.dataset:
            img = self.transform(img).to(dev_str)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            self.inputs.append((path, img, img_det, vid_cap, shapes))
    
    def launch_gpu_work(self):
        conf_thres = 0.25
        iou_thres = 0.45
        img_size = 640
        for path, img, img_det, vid_cap,shapes in self.inputs:
            det_out, da_seg_out,ll_seg_out= self.model(img)

            inf_out, _ = det_out
            det_pred = non_max_suppression(
                inf_out,
                conf_thres=conf_thres, 
                iou_thres=iou_thres,
                classes=None, agnostic=False
            )
            det=det_pred[0]

            _, _, height, width = img.shape
            h,w,_ = img_det.shape
            pad_w, pad_h = shapes[1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[1][0][1]

            da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
            da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze()
            # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
            
            ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
            ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze()
        
        return da_seg_mask, ll_seg_mask

TASKNAME_TO_CLASS = {
    'alexnet': AlexnetTest,
    'resnet50': Resnet50Test,
    'bertsquad': BertSquadTest,
    'vgg19': Vgg19Test,
    'yolop': YolopTest
}

def run_concurrent_tests(num_instances, gpu_ids, TASKNAME, **kwargs):
    tests = []
    task_class = TASKNAME_TO_CLASS.get(TASKNAME)
    for gpu_id in gpu_ids:
        test = task_class(gpu_id, **kwargs)
        tests.append(test)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_instances) as executor:
        future_to_test = {executor.submit(test.run_bench): i 
                         for i, test in enumerate(tests)}
        
        for future in concurrent.futures.as_completed(future_to_test):
            instance_id = future_to_test[future]
            results.append((instance_id, future.result()))
    
    return sorted(results, key=lambda x: x[0])

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mig', 'mps', 'mps_limit_sm'], required=True)
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--task_name', choices=TASKNAME_TO_CLASS.keys(), required=True)
    args = parser.parse_args()
    
    manager = GPUInstanceManager(args.mode)
    try:
        if args.mode == 'mig':
            manager.setup_mig(args.num_instances)
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            gpu_ids = [x for x in re.findall(r'(MIG-[a-f0-9-]+)', result.stdout)]
        else:
            manager.setup_mps(args.num_instances)
            # For MPS, we just use sequential IDs
            gpu_ids = list(range(args.num_instances))
        
        results = run_concurrent_tests(args.num_instances, gpu_ids, args.task_name)
        
        print(f"\nResults for {args.mode.upper()} with {args.num_instances} instances:")
        print("-" * 50)
        print(f"Task: {args.task_name}")
        for instance_id, result in results:
            print(f"Instance {instance_id}:")
            print(f"Mean latency: {np.mean(result['latencies']):.3f}s")
            print(f"Std deviation: {np.std(result['latencies']):.3f}s")
            print(f"P99 latency: {np.percentile(result['latencies'], 99):.3f}s")
            print()
        
        # append results to a file
        if not os.path.isfile("results_mlinfer.csv"):
            with open("results_mlinfer.csv", "w") as f:
                f.write("task_name,mode,num_instances,instance_id,latency\n")
        with open('results_mlinfer.csv', 'a') as f:
            for instance_id, result in results:
                for latency in result['latencies']:
                    f.write(f"{args.task_name},{args.mode},{args.num_instances},{instance_id},{latency}\n")
            
    finally:
        manager.cleanup()
