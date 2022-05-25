import torch
import torch.quantization
from torch import cuda
from torch import nn, from_numpy, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os

print(torch.__version__)
device = 'cpu'


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■ setting ■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

class Test_Dataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data_/test_freezed.csv',delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, :-1])
        # scaler.fit(self.x_data)
        # self.x_data = scaler.transform(self.x_data)
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

test_dataset = Test_Dataset()
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=len(test_dataset),
                         shuffle=True)


class Model(nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.l1 = nn.Linear(10, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 4)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # DeQuantStub converts tensors from quantized to floating point

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x


def print_size_of_model(model, label=""):
    # torch.save(model.state_dict(), "temp_%s.pt" %(label))
    torch.save(model.state_dict(), "quantized_train_model/model_%s.pt" % (label))
    size=os.path.getsize("quantized_train_model/model_%s.pt" %(label))
    print("model: ",label,' \t','Size (KB):', size/1e3)
    # os.remove("temp/temp_%s.pt" %(label))

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return size



def confusion_mtrix_plot(ytrue, ypred, name):
    confmat = confusion_matrix(y_true=ytrue, y_pred=ypred)
    sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues')
    plt.title('supervised classifier')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig('quantized_test_results/%s_confusion_matrix.png' %(name))
    plt.close()
    with open("quantized_test_results/%s_classification_report.txt" %(name), "w") as text_file:
        print(classification_report(ytrue, ypred, digits=4, target_names=['0', '1', '2', '3']),
              file=text_file)


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■ Dynamic Quantization ■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
'''
■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
■■■■■■■ Dynamic Quantization ■■■■■■■■
■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# https://velog.io/@jooh95/%EB%94%A5%EB%9F%AC%EB%8B%9D-Quantization%EC%96%91%EC%9E%90%ED%99%94-%EC%A0%95%EB%A6%AC
+ 가장 간단한 양자화 기법
+ 모델의 가중치(weight)에 대해서만 양자화 진행
+ 활성화(activations)는 추론할 때 동적으로 양자화
    + activations는 메모리에 부동소수점 형태로 read, write 됨.
    + inference시에만 floating-point kernel를 이용해여 weights를 int8을 float32로 convert됨. 
      Activation은 항상 floating point로 저장되어져 있습니다. 그래서 quantized kernels processing을 지원하는 operator의 경우에는 
      activation을 processing전에 dynamic하게 8bit로 quantized하고 processing후에 다시 dequantization하게 됩니다.
    + weights들은 training 후에 quantize
    + activations은 inference time에 dynamic하게 quantized
+ 모델을 메모리에 로딩하는 속도 개선
+ 연산속도 향상 효과 미비 (inference kernel 연산이 필요하기 때문)
+ CPU 환경에서만 inference 가능 (프레임워크나 프레임워크의 버전에따라 GPU 환경에서도 동작할 순 있음)
    + With PyTorch 1.7.0, we could do dynamic quantization using x86-64 and aarch64 CPUs. 
      However, NVIDIA GPUs have not been supported for PyTorch dynamic quantization yet.
+ 모델의 weights를 메모리에 loading하는 것이 execution time에 큰 영향을 미치는 BERT와 같은 모델에 적합
+ CPU 환경에서만 inference 가능 (프레임워크나 프레임워크의 버전에따라 GPU 환경에서도 동작할 순 있음)
+ 모델의 weights를 메모리에 loading하는 것이 execution time에 큰 영향을 미치는 BERT와 같은 모델에 적합
'''
model_fp32 = Model()
model_fp32 = torch.load('quantized_train_model/trained_model.pt')
model_fp32 = model_fp32.to(device)

model_int8  = torch.quantization.quantize_dynamic(model_fp32.to(device), {torch.nn.Linear}, dtype=torch.qint8)
# torch.save(model_int8, 'quantized_train_model/dyna_qat_model_int8.pt')

print('■■■■■■■■ ORGIN ■■■■■■■■')
print(model_fp32)
f = print_size_of_model(model_fp32,"fp32") # Model size

print('\n\n■■■■■■■■ DYNA INT8 ■■■■■■■■')
print(model_int8)
q = print_size_of_model(model_int8,"int8") # Model size
for param in model_int8.parameters():
    print(param)

print('\n\n■■■■■■■■ SIZE COMPARISON ■■■■■■■■')
print("{0:.2f} times smaller".format(f/q))

# ■■■■■■■■■■ test ■■■■■■■■■■


test_loader_all = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
for batch_idx, (inputs_all, labels_all) in enumerate(test_loader_all):
    inputs_all, labels_all = inputs_all.to(device=device).float(), labels_all.to(device=device).float()
    
    y_pred_all_int8 = model_int8(inputs_all).to(device)
    y_pred_all_int8 = y_pred_all_int8.data.max(1, keepdim=True)[1]
    y_pred_all_int8 = y_pred_all_int8.cpu().detach().numpy()

    y_pred_all_fp32 = model_fp32(inputs_all).to(device)
    y_pred_all_fp32 = y_pred_all_fp32.data.max(1, keepdim=True)[1]
    y_pred_all_fp32 = y_pred_all_fp32.cpu().detach().numpy()

    labels_all = labels_all.cpu().detach().numpy()


confusion_mtrix_plot(labels_all, y_pred_all_fp32, 'org_fp32')
confusion_mtrix_plot(labels_all, y_pred_all_int8, 'dyna_qat_int8')

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■