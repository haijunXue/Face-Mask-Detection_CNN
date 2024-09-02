import torch

print(torch.__version__)
print('gpu', torch.cuda.is_available())

# minimize loss
# loss = sum(w*xi + b -y)*2
#convex optimization

#Linear Regression连续函数
# Linear Regression[-...,+....]
# Logistic Regression[0,1]
#Classification分类问题
