import pandas as pd

df = pd.read_csv('../../data/workload_share/rodinia.csv')

avg = df.groupby(['test']).mean()

print('-----Native------')
print(avg['native fulltime'])
print(avg['native (kernel+cudaMemcpy)'] / avg['native fulltime'] * 100)
print(avg['native (kernel+cudaMemcpy+cudaMalloc+cudaFree)'] / avg['native fulltime'] * 100)

print('-----TCP------')
print(avg['gpuless fulltime'])
print(avg['gpuless (kernel+cudaMemcpy)'] / avg['gpuless fulltime'] * 100)
print(avg['gpuless (kernel+cudaMemcpy+cudaMalloc+cudaFree)'] / avg['gpuless fulltime'] * 100)
