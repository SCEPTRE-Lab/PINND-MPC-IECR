import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import pi
import random
from sklearn.utils import shuffle
import pandas as pd

randnum = 3134
random.seed(randnum)
torch.random.seed = randnum
torch.seed = randnum
np.random.seed(randnum)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iterations = 500000
no_neurons = 64
bestmodel = 'SOFC_ODE_both_wbias.pth'
alpha_ic = 1
alpha_p = 1
alpha_d = 1
final_simulation_time = 10

# Parameters
Temp = 1273
Eo = 1.18
Uf = 0.8
rHO = 1.145
F = 96486
N = 280
R = 8.314
iL = 800
alpha = 0.2
beta = -2870
To = 973
A = 101.2e3
Eact = 120e3

tau_h2 = 26.1
tau_o2 = 29.1
tau_h2o = 78.3

KH2 = 0.843
KO2 = 2.52
KH2O = 0.281
Kr = 0.993e-3

t_data = np.linspace(0, 10, 100)
Ifc0 = 175
QH2in_0 = 0.5


def sofc_model(y, t, params):
    N = 280#-5
    KH2O = 0.281#+0.04
    ph2, po2, ph2o = y
    Ifc = params[0]
    QH2in_0 = params[1]
    QO2in_0 = QH2in_0 / rHO
    QH2Oin_0 = 0

    dh2dt = (1 / (tau_h2 * KH2)) * (QH2in_0 - KH2 * ph2 - (Ifc * N) / (2 * F))
    do2dt = (1 / (tau_o2 * KO2)) * (QO2in_0 - KO2 * po2 - 0.5 * (Ifc * N) / (2 * F))
    dh2odt = (1 / (tau_h2o * KH2O)) * (QH2Oin_0 - KH2O * ph2o + (Ifc * N) / (2 * F))
    dydt = [dh2dt, do2dt, dh2odt]
    return dydt


# sol = odeint(sofc_model, np.array([0, 0, 0]), t_data, args=(np.array([Ifc0,QH2in_0]),))


class SaveBestModel:

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            # print(f"\nBest validation loss: {self.best_valid_loss}")
            # print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, bestmodel)


def save_model(epochs, model, optimizer, criterion):
    filename = 'int_both_' + str(epochs) + '.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, filename)


save_best_model = SaveBestModel()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1 + 2 + 3, no_neurons)
        self.hidden_layer2 = nn.Linear(no_neurons, no_neurons)
        self.hidden_layer3 = nn.Linear(no_neurons, no_neurons)
        self.hidden_layer4 = nn.Linear(no_neurons, no_neurons)
        self.hidden_layer5 = nn.Linear(no_neurons, no_neurons)
        self.output_layer = nn.Linear(no_neurons, 3)

    def forward(self, t, Ifc_in, QH2IN, ic1, ic2, ic3):
        t = (t - 0) / final_simulation_time
        Ifc_in = (Ifc_in - 100) / 600
        QH2IN = (QH2IN - 0) / 2
        ic1 = (ic1 - 0) / 1.5
        ic2 = (ic2 - 0) / 1.5
        ic3 = (ic3 - 0) / 4

        inputs = torch.cat([t, Ifc_in, QH2IN, ic1, ic2, ic3], axis=1)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        output_temp = self.output_layer(layer5_out)
        return output_temp


''' (2) LOADING THE MODEL AND OPTIMISERS'''
# torch.optim.LBFGS(N.parameters())
net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()
lr = 0.002
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def f(t, Ifc, QH2in, ph2, po2, ph2o, net):
    N = 280-5
    KH2O = 0.281+0.04

    output = net(t, Ifc, QH2in, ph2, po2, ph2o)

    ph2 = output[:, 0].reshape([output.shape[0], 1])
    po2 = output[:, 1].reshape([output.shape[0], 1])
    ph2o = output[:, 2].reshape([output.shape[0], 1])

    # Calculating the derivatives
    ph2_t = torch.autograd.grad(ph2.sum(), t, create_graph=True)[0]
    po2_t = torch.autograd.grad(po2.sum(), t, create_graph=True)[0]
    ph2o_t = torch.autograd.grad(ph2o.sum(), t, create_graph=True)[0]

    de1 = ph2_t - (1 / (tau_h2 * KH2)) * (QH2in - KH2 * ph2 - (Ifc * N) / (2 * F))
    de2 = po2_t - (1 / (tau_o2 * KO2)) * (QH2in / rHO - KO2 * po2 - 0.5 * (Ifc * N) / (2 * F))
    de3 = ph2o_t - (1 / (tau_h2o * KH2O)) * (0 - KH2O * ph2o + (Ifc * N) / (2 * F))

    return de1, de2, de3


# (3) Training the NN Model
previous_validation_loss = 99999999.0

trainingEpoch_loss = []
trainingEpoch_pdeloss = []
trainingEpoch_icloss = []
trainingEpoch_mseval = []

ic_collocation1 = shuffle(
    np.tile(np.concatenate([np.random.uniform(low=0.0, high=1.5, size=(80, 1)), np.zeros(20).reshape(20, 1)]),
            100).reshape(100 * 100, 1))
ic_collocation2 = shuffle(
    np.tile(np.concatenate([np.random.uniform(low=0.0, high=1.5, size=(80, 1)), np.zeros(20).reshape(20, 1)]),
            100).reshape(100 * 100, 1))
ic_collocation3 = shuffle(
    np.tile(np.concatenate([np.random.uniform(low=0.0, high=4, size=(80, 1)), np.zeros(20).reshape(20, 1)]),
            100).reshape(100 * 100, 1))
Q_collocation = shuffle(np.tile(np.random.uniform(low=0.0, high=2, size=(100, 1)), 100).reshape(100 * 100, 1))
t_mesh, I_mesh = np.meshgrid(np.linspace(0, final_simulation_time, 100), np.linspace(100, 700, 100))

ic1_data = ic_collocation1
ic2_data = ic_collocation2
ic3_data = ic_collocation3
Q_data = Q_collocation
t_data = t_mesh.flatten().reshape(100 * 100, 1)
Ifc_data = I_mesh.flatten().reshape(100 * 100, 1)
inival_array = np.zeros([5, 100 * 100])

Ntd = 10



def gen_traindata():
    mvv_1 = Ifc_data
    mvv_2 = Q_data
    ic_1 = ic1_data
    ic_2 = ic2_data
    ic_3 = ic3_data

    tt = np.linspace(0, final_simulation_time, Ntd)
    ones = np.ones(Ntd).reshape([Ntd, 1])
    Final_input_data = pd.DataFrame()
    Final_output_data = pd.DataFrame()

    for i in range(len(mvv_1)):
        y0 = np.array([ic_1[i], ic_2[i], ic_3[i]])
        mvv = np.array([mvv_1[i], mvv_2[i]]).ravel()
        sol = pd.DataFrame(odeint(sofc_model, y0.ravel(), tt.ravel(), args=(mvv,)))

        inp = np.concatenate(
            [tt.reshape([Ntd, 1]), ones * mvv_1[i], ones * mvv_2[i], ones * ic_1[i], ones * ic_2[i],
             ones * ic_3[i]], axis=1)
        Final_input_data = pd.concat([Final_input_data, pd.DataFrame(inp)], axis=0)
        Final_output_data = pd.concat([Final_output_data, pd.DataFrame(sol)], axis=0)

    return Final_input_data, Final_output_data


Train_data = gen_traindata()
Final_input_data = Train_data[0]
Final_output_data2 = Train_data[1]

Final_output_data = Final_output_data2 + np.random.uniform(-0.05, 0.05, size=Final_output_data2.shape)*Final_output_data2 #- [0.01,0.02,0.1]
Npinn = 100*100
Ndata = 100*100*10

inp_t = Variable(torch.from_numpy(t_data).float(), requires_grad=True).to(device)
inp_mv1 = Variable(torch.from_numpy(Ifc_data).float(), requires_grad=False).to(device)
inp_mv2 = Variable(torch.from_numpy(Q_data).float(), requires_grad=False).to(device)
pt_ic1 = Variable(torch.from_numpy(ic1_data).float(), requires_grad=False).to(device)
pt_ic2 = Variable(torch.from_numpy(ic2_data).float(), requires_grad=False).to(device)
pt_ic3 = Variable(torch.from_numpy(ic3_data).float(), requires_grad=False).to(device)
pt_zeros = Variable(torch.from_numpy(np.zeros(Npinn).reshape([Npinn, 1])).float(), requires_grad=False).to(device)

ip_inp_t = Variable(torch.from_numpy(Final_input_data.iloc[:, 0].to_numpy().reshape([Ndata, 1])).float(),
                    requires_grad=True).to(device)
ip_inp_mv1 = Variable(torch.from_numpy(Final_input_data.iloc[:, 1].to_numpy().reshape([Ndata, 1])).float(),
                      requires_grad=False).to(device)
ip_inp_mv2 = Variable(torch.from_numpy(Final_input_data.iloc[:, 2].to_numpy().reshape([Ndata, 1])).float(),
                      requires_grad=False).to(device)
ip_pt_ic1 = Variable(torch.from_numpy(Final_input_data.iloc[:, 3].to_numpy().reshape([Ndata, 1])).float(),
                     requires_grad=False).to(device)
ip_pt_ic2 = Variable(torch.from_numpy(Final_input_data.iloc[:, 4].to_numpy().reshape([Ndata, 1])).float(),
                     requires_grad=False).to(device)
ip_pt_ic3 = Variable(torch.from_numpy(Final_input_data.iloc[:, 5].to_numpy().reshape([Ndata, 1])).float(),
                     requires_grad=False).to(device)

ip_pt_zeros = Variable(torch.from_numpy(np.zeros(Ndata).reshape([Ndata, 1])).float(), requires_grad=True).to(device)

op_pt_ic1 = Variable(torch.from_numpy(Final_output_data.iloc[:, 0].to_numpy().reshape([Ndata, 1])).float(),
                     requires_grad=False).to(device)
op_pt_ic2 = Variable(torch.from_numpy(Final_output_data.iloc[:, 1].to_numpy().reshape([Ndata, 1])).float(),
                     requires_grad=False).to(device)
op_pt_ic3 = Variable(torch.from_numpy(Final_output_data.iloc[:, 2].to_numpy().reshape([Ndata, 1])).float(),
                     requires_grad=False).to(device)

min_valid_loss = np.inf

# best_model_cp = torch.load(bestmodel)
# net.load_state_dict(best_model_cp['model_state_dict'])

for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero
    # if epoch % int(iterations / 10) == 0:
    #     save_model(epoch, net, optimizer, mse_cost_function)

    # INITIAL CONDITIONS
    net_ic_out0 = net(pt_zeros, inp_mv1, inp_mv2, pt_ic1, pt_ic2, pt_ic3)
    mse_ic_1 = mse_cost_function(net_ic_out0[:, 0].reshape([net_ic_out0.shape[0], 1]), pt_ic1)
    mse_ic_2 = mse_cost_function(net_ic_out0[:, 1].reshape([net_ic_out0.shape[0], 1]), pt_ic2)
    mse_ic_3 = mse_cost_function(net_ic_out0[:, 2].reshape([net_ic_out0.shape[0], 1]), pt_ic3)
    mse_ic = mse_ic_1 + mse_ic_2 + mse_ic_3

    # # # # # PDE RESIDUALS IN DOMAIN
    f_out = f(inp_t, inp_mv1, inp_mv2, pt_ic1, pt_ic2, pt_ic3, net)
    mse_f1 = mse_cost_function(f_out[0], pt_zeros)
    mse_f2 = mse_cost_function(f_out[1], pt_zeros)
    mse_f3 = mse_cost_function(f_out[2], pt_zeros)
    mse_f = mse_f1 + mse_f2 + mse_f3

    net_ic_outd = net(ip_inp_t, ip_inp_mv1, ip_inp_mv2, ip_pt_ic1, ip_pt_ic2, ip_pt_ic3)
    mse_d_1 = mse_cost_function(net_ic_outd[:, 0].reshape([net_ic_outd.shape[0], 1]), op_pt_ic1)
    mse_d_2 = mse_cost_function(net_ic_outd[:, 1].reshape([net_ic_outd.shape[0], 1]), op_pt_ic2)
    mse_d_3 = mse_cost_function(net_ic_outd[:, 2].reshape([net_ic_outd.shape[0], 1]), op_pt_ic3)
    mse_d = mse_d_1 + mse_d_2 + mse_d_3

    # COMBINING THE LOSS FUNCTION
    loss = alpha_ic * mse_ic + alpha_p * mse_f + alpha_d * mse_d
    loss.backward()
    optimizer.step()

    save_best_model(loss, epoch, net, optimizer, mse_cost_function)

    with torch.autograd.no_grad():
        print(epoch, "Training Loss: ", loss.item(), " function: ", mse_f.item(), " IC: ", mse_ic.item(), "Data: ",
              mse_d.item(), 'lr = ', lr)
    #
    if epoch % 50000 == 0:
        lr = lr / 2
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    trainingEpoch_loss.append(np.array(loss.data.cpu()))
    trainingEpoch_pdeloss.append(np.array(mse_f.data.cpu()))
    trainingEpoch_icloss.append(np.array(mse_ic.data.cpu()))
    trainingEpoch_mseval.append(np.array(mse_d.data.cpu()))

best_model_cp = torch.load(bestmodel)
net.load_state_dict(best_model_cp['model_state_dict'])
test = Final_input_data.iloc[40000,:]
Ifc0 = test[1]
QH2in_0 = test[2]
t = np.linspace(0, final_simulation_time, Ntd)
pt_t = Variable(torch.from_numpy(t.reshape(-1, 1)).float(), requires_grad=False).to(device)
pt_u = net(pt_t, pt_t * 0 + Ifc0, pt_t * 0 + QH2in_0, pt_t * 0 + test[3], pt_t * 0 + test[4], pt_t * 0 + test[5])
pred = pt_u.cpu().detach().numpy()

y0 = np.array([test[3], test[4], test[5]])
sol = odeint(sofc_model, y0, t.ravel(), args=(np.array([Ifc0,QH2in_0]),))
#
epoch_arr = np.linspace(0, iterations, iterations)
plt.figure(2, figsize=[8,6])
plt.plot(epoch_arr, trainingEpoch_icloss, '--', label='IC loss')
plt.plot(epoch_arr, trainingEpoch_pdeloss, label='Physics-based loss')
plt.plot(epoch_arr,trainingEpoch_mseval, label='Datadriven loss')
plt.plot(epoch_arr, trainingEpoch_loss, '--', label='Total loss')
plt.title('Pure PINN model loss curves')
plt.xlabel('No. of training epochs')
plt.ylabel('MSE loss values')
plt.legend()
plt.grid()
plt.semilogy()
# plt.savefig('Data_Model.eps', format='eps')
# plt.savefig('Data_Model.svg', format='svg', dpi=1200)
# plt.savefig('Data_Model.png', format='jpg', dpi=1200)
# Loss_dataset = pd.DataFrame([np.array(trainingEpoch_pdeloss),
#                              np.array(trainingEpoch_mseval), np.array(trainingEpoch_loss),
#                              np.array(trainingEpoch_icloss)]).to_csv('Both_loss.csv')

def plot_fig():
    plt.figure(1)
    plt.plot(t, pred[:, 0], '--', label='Ca at x = 0.25')
    plt.plot(t, pred[:, 1], '--', label='Ca at x = 0.50')
    plt.plot(t, pred[:, 2], '--', label='Ca at x = 0.75')

    plt.plot(t, sol[:, 0], label='Ca at x = 0.25')
    plt.plot(t, sol[:, 1], label='Ca at x = 0.50')
    plt.plot(t, sol[:, 2], label='Ca at x = 0.75')
    plt.legend()

plot_fig()
