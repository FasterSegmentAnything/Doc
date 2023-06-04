import time
import torch
import onnxruntime

##################################################
print("pytorch version:`%s`"%(torch.__version__))
print("onnxruntime version:`%s`"%(onnxruntime.__version__))

for i in range(1,10):
    start = time.time()
    a = torch.FloatTensor(i*10,10,10)
    a = a.cuda() #a = a
    a = torch.matmul(a,a)
    end = time.time() - start
    print("run(%d) cost: %f"%(i,end))