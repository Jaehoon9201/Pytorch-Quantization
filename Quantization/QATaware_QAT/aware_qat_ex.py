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
import copy
import time
from sklearn.preprocessing import StandardScaler

print(torch.__version__)
device = 'cpu'



# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■ setting ■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
Epoch_num = 40
batch_size = 64
train_loss_values = np.zeros([Epoch_num - 1])
test_loss_values = np.zeros([Epoch_num - 1])
scaler = StandardScaler()

class Train_Dataset(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, downlo ad, etc.
    def __init__(self):
        xy = np.loadtxt('./data_/train_freezed.csv',delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]

        self.x_data = from_numpy(xy[:, :-1])
        # scaler.fit(self.x_data)
        # self.x_data  = scaler.transform(self.x_data)
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


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

train_dataset = Train_Dataset()
test_dataset = Test_Dataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)


class Model_Fp32(nn.Module):

    def __init__(self, org_model):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model_Fp32, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.l1 = nn.Linear(10, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 4)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.org_model = org_model
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        x = self.quant(x)
        x = self.org_model(x)
        x = self.dequant(x)
        return x


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
    torch.save(model.state_dict(), "quantized_train_model/model_%s.pt" % (label))
    size = os.path.getsize("quantized_train_model/model_%s.pt" % (label))
    print("model: ",label,' \t','Size (KB):', size/1e3)
    # os.remove("temp/temp_%s.pt" %(label))

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

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


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■ Static Quantization ■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
'''
■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
■■■■■■■ Static Quantization ■■■■■■■■■
■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# https://velog.io/@jooh95/%EB%94%A5%EB%9F%AC%EB%8B%9D-Quantization%EC%96%91%EC%9E%90%ED%99%94-%EC%A0%95%EB%A6%AC
+ 모델의 가중치와 활성화(activations) 모두 양자화를 사전에 진행
    + 가중치와 활성화 fusion
    + calibration하는 동안 활성화가 설정됨
        * calibration: 직역하면 눈금 매김 (미세조정 같은 것으로 이해)
+ static quantization의 활성화 quantization을 위해 activation의 preceding layer와 fusion 수행
    - fusion = 각각의 기능을 수행하는 layer를 하나로 합침.
    - activation, convolution 등 layer를 합쳐 layer 간에 데이터 이동으로 발생하는 context switching overhead를 줄일 수 있음.
    - sequential하게 처리되던 연산을 병렬도 처리할 수 있다는 장점도 있음.
    - [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu] 등과 같은 fusion이 있음.
+ 정확도 손실을 최소화하기 위해 calibration으로 미세조정
    + calibration으로 range 설정을 조절할 수 있음.
    + histogram에따른 calibration 등
    + 대표 데이터셋을 통해 calibration 조정
+ 연산속도 향상
+ tflite는 CPU, GPU 환경에서 추론 가능 / pytorch는 CPU만 가능

#https://pytorch.org/docs/stable/quantization.html

'''


model = Model()
model = torch.load('quantized_train_model/trained_model.pt')
model_fp32 = Model_Fp32(model)
model_fp32 = model_fp32.to(device)
model_fp32 = model_fp32.train()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
# model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['l1', 'relu1'],  ['l2', 'relu2',]], inplace=True)  # ['l3']

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■        TRAINING        ■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fp32_prepared.parameters(), lr=0.002)

def train(epoch, train_loss_values):
    train_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # get the inputs
        inputs, labels = inputs.to(device=device).float(), labels.to(device=device).float()


        #e Forward pass: Compute predicted y by passing x to the model
        optimizer.zero_grad()
        y_pred = model_fp32_prepared(inputs)
        labels = torch.reshape(labels, [-1]).to(device)

        loss = criterion(y_pred, labels.long())
        train_loss +=loss
        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print(train_loss)
    train_loss /= len(train_loader.dataset)
    #train_loss_values.append(loss / len(inputs))
    #train_loss_values = np.append(train_loss_values, train_loss)
    train_loss_values[epoch-1] = train_loss
    model_int8 = model_fp32_prepared
    #torch.save(model_int8, 'quantized_train_model/model_int8.pt')
    return train_loss_values, model_int8



for epoch in range(1, Epoch_num):
    epoch_start = time.time()
    train_loss_values, model_int8 = train(epoch, train_loss_values)
    m, s = divmod(time.time() - epoch_start, 60)
    print(f'Training time: {m:.0f}m {s:.0f}s')
    # test_loss_values = test(test_loss_values)
    # m, s = divmod(time.time() - epoch_start, 60)
    # print(f'Testing time: {m:.0f}m {s:.0f}s')

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8.eval()
model_int8 = torch.quantization.convert(model_int8)


print('■■■■■■■■ ORGIN ■■■■■■■■')
print(model)
f = print_size_of_model(model,"fp32") # Model size

print('\n\n■■■■■■■■ DYNA INT8 ■■■■■■■■')
print(model_int8)
q = print_size_of_model(model_int8,"int8") # Model size

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
confusion_mtrix_plot(labels_all, y_pred_all_int8, 'aware_qat_int8')

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■