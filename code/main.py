import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets


from scipy.sparse.csgraph import laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, index_to_mask
from torch_geometric.nn import GCNConv

from EarlyStopping import EarlyStopping

import networkx as nx

import warnings

import pickle

#%% Setting

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

font = font_manager.FontProperties(family='Cambria',
                                   style='normal', size=13)


#%% Function

def graph_construct(data, sigma = 1, neighbor = 30, penalty = True):
    dist = pairwise_distances(data, metric = 'euclidean')
    e = np.mean(dist)
    W_temp = np.exp(-(dist **2) / (sigma * e))
    
    A = kneighbors_graph(data, neighbor, mode='connectivity', include_self = False)
    A = 0.5 * (A + A.T)

    if not penalty:
        A[A == 0.5] = 1
    
    A = A.toarray()
    W = W_temp * A
    
    return W


def labeled_Laplacian(L, labeled):
    L_labeled = L[labeled][:, labeled]
    
    return L_labeled 


def SGF(L, y, labeled):

    L_labeled = []
    num_graph = L.shape[0]
    num_node = L[0].shape[0]
    y = y.copy()
    
    
    for i in range(num_graph):
        L_labeled.append(L[i])
    
    L_labeled = np.array(L_labeled)
    
    trm = np.zeros((num_graph, num_graph))
    for i in range(num_graph - 1):
        for j in range(i + 1, num_graph):
            trm[i, j] = (L_labeled[i] @ L_labeled[j]).diagonal().sum()
            
    trm = trm + trm.T
    
    for i in range(num_graph):
        trm[i, i] = (L_labeled[i] @ L_labeled[i]).diagonal().sum()
    
    trm = csc_matrix(trm)
    
    smoothness = np.zeros((num_graph, 1))
    for i in range(num_graph):
        smoothness[i] = num_node - (y.T @ L_labeled[i] @ y)
    smoothness = csc_matrix(smoothness)
    
    alpha = spsolve(trm, smoothness)
    return alpha 


def softmax(alpha):
    return np.exp(alpha)/sum(np.exp(alpha))


def graph_integration(matrix, alpha):
    
    num_graph = matrix.shape[0]
    num_node = matrix[0].shape[1]
    
    M_integration = np.zeros((num_node, num_node))
    for i in range(num_graph):
        M_integration = M_integration + alpha[i]*matrix[i]
        
    return csc_matrix(M_integration)


def degree_matrix(A):
    return np.diag(A.sum(axis = 1))


def transition_matrix_sym(A, w_loop = 0):
    N = A.shape[0]
    D = degree_matrix(A)
    A_tild = w_loop*np.eye(N) + A
    D_tild = w_loop*np.eye(N) + D
    D_inv_sqrt = np.linalg.inv(np.sqrt(D_tild))
    return D_inv_sqrt @ A_tild @ D_inv_sqrt


def page_rank(a, k):
    theta = []
    for i in range(k+1):
        theta_k = a*(1-a)**i
        theta.append(theta_k)
    return theta


def graph_diffusion(T, theta):
    N = T.shape[0]
    S = np.zeros((N, N))
    
    for k, theta_k in enumerate(theta):
        S = S + theta_k * np.linalg.matrix_power(T, k)
    
    return S



def similarity_kernel(X, mu = 0.5):
    n = X.shape[0]
    
    dist = pairwise_distances(X, metric = 'euclidean')
    e = np.mean(dist, axis = 0).reshape((n , 1))
    e = (e + e.T + dist)/3 

    W = np.exp(-(dist**2) / (mu*e))
    
    return W


def normalized_weight_matrix(W, normalization = True):
    
    if normalization:
        W = W.copy()
        np.fill_diagonal(W, 0)
        D = degree_matrix(W)
        P = (np.linalg.inv(D) @ W) / 2
        np.fill_diagonal(P, 1/2)
    
    else: 
        D = degree_matrix(W)
        P = np.linalg.inv(D) @ W
    
    return P


def SNF(X_li, S_li, num_iter):
    
    num_graph = len(X_li)
    num_node = X_li[0].shape[0]
    
    P_total = [normalized_weight_matrix(similarity_kernel(X)) for X in X_li]
    S_total = [S.toarray() + np.eye(num_node) for S in S_li]
    
    P_new = P_total.copy()
    
    for itr in range(num_iter):
        
        P_t = P_new.copy()
        
        for i in range(num_graph):
            P_sum = np.zeros((num_node, num_node))
            for j in range(num_graph):
                if i == j:
                    continue
                P_sum = P_sum + P_t[j]
            P_sum = P_sum / (num_graph - 1)
            P_new[i] = S_total[i] @ P_sum @ S_total[i].T
            
        for i in range(num_graph):
            P_temp = P_new[i].copy()
            np.fill_diagonal(P_temp, 0)
            D = np.diag(np.sum(P_temp, axis = 1))
            P_norm = (np.linalg.inv(D) @ P_temp) / 2
            np.fill_diagonal(P_norm, 1/2)
            P_new[i] = P_norm
            
            
    P_int = np.zeros((num_node, num_node))
    
    for i in range(num_graph):
        P_int = P_int + P_new[i]
    
    P_int = P_int / num_graph
    P_int = (P_int + P_int.T) / 2
    
    return P_int



def multiplex_Laplacian(W_total):
    
    L_new = []
    for W in W_total:
        L_new.append(laplacian(W, normed = True) + 0)
        
    return np.array(L_new)
   

def multiplex_Laplacian_csc(W_total):
    
    L_new = []
    for W in W_total:
        L_new.append(csc_matrix(laplacian(W, normed = True) + 0))
        
    return np.array(L_new)
    

def SNF_star(X_li, S_li, y_ssl, labeled, num_iter):
    
    num_graph = len(X_li)
    num_node = X_li[0].shape[0]
    
    P_total = [normalized_weight_matrix(similarity_kernel(X)) for X in X_li]
    S_total = [S.toarray() + np.eye(num_node) for S in S_li]
    L_total = multiplex_Laplacian(S_total)
    alpha = SGF(L_total, y_ssl, labeled)
    alpha = softmax(alpha)
    P_new = P_total.copy()
    
    for itr in range(num_iter):
        
        P_t = P_new.copy()
        
        for i in range(num_graph):
            P_sum = np.zeros((num_node, num_node))
            for j in range(num_graph):
                if i == j:
                    continue
                P_sum = P_sum + P_t[j]
            P_sum = P_sum / (num_graph - 1)
            P_new[i] = S_total[i] @ P_sum @ S_total[i].T
            
        for i in range(num_graph):
            P_temp = P_new[i].copy()
            np.fill_diagonal(P_temp, 0)
            D = np.diag(np.sum(P_temp, axis = 1))
            P_norm = (np.linalg.inv(D) @ P_temp) / 2
            np.fill_diagonal(P_norm, 1/2)
            P_new[i] = P_norm
            
    
    P_int = graph_integration(np.array(P_new), alpha)
    
    return P_int


def SNF_P(X_li, S_li, y_ssl, labeled, num_iter):
    
    num_graph = len(X_li)
    num_node = X_li[0].shape[0]
    
    P_total = [normalized_weight_matrix(similarity_kernel(X)) for X in X_li]
    S_total = [S.toarray() + np.eye(num_node) for S in S_li]
    L_total = multiplex_Laplacian(S_total)
    alpha = SGF(L_total, y_ssl, labeled)
    P_new = P_total.copy()
    
    for itr in range(num_iter):
        
        P_t = P_new.copy()
        
        for i in range(num_graph):
            P_sum = np.zeros((num_node, num_node))
            alpha_current = alpha.copy()
            alpha_current[i] = -np.inf
            alpha_current = softmax(alpha_current)
            
            for j in range(num_graph):
                if i == j:
                    continue
                P_sum = P_sum + alpha_current[j]*P_t[j]
            
            P_new[i] = S_total[i] @ P_sum @ S_total[i].T
            
        for i in range(num_graph):
            P_temp = P_new[i].copy()
            np.fill_diagonal(P_temp, 0)
            D = np.diag(np.sum(P_temp, axis = 1))
            P_norm = (np.linalg.inv(D) @ P_temp) / 2
            np.fill_diagonal(P_norm, 1/2)
            P_new[i] = P_norm
            
    
    P_int = np.zeros((num_node, num_node))
    
    for i in range(num_graph):
        P_int = P_int + P_new[i]
    
    P_int = P_int / num_graph
    P_int = (P_int + P_int.T) / 2
    
    return P_int


def SNF_P_star(X_li, S_li, y_ssl, labeled, num_iter):
    
    num_graph = len(X_li)
    num_node = X_li[0].shape[0]
    
    P_total = [normalized_weight_matrix(similarity_kernel(X)) for X in X_li]
    S_total = [S.toarray() + np.eye(num_node) for S in S_li]
    L_total = multiplex_Laplacian(S_total)
    alpha = SGF(L_total, y_ssl, labeled)
    P_new = P_total.copy()
    
    for itr in range(num_iter):
        
        P_t = P_new.copy()
        
        for i in range(num_graph):
            P_sum = np.zeros((num_node, num_node))
            alpha_current = alpha.copy()
            alpha_current[i] = -np.inf
            alpha_current = softmax(alpha_current)
            
            for j in range(num_graph):
                if i == j:
                    continue
                P_sum = P_sum + alpha_current[j]*P_t[j]
            
            P_new[i] = S_total[i] @ P_sum @ S_total[i].T
            
        for i in range(num_graph):
            P_temp = P_new[i].copy()
            np.fill_diagonal(P_temp, 0)
            D = np.diag(np.sum(P_temp, axis = 1))
            P_norm = (np.linalg.inv(D) @ P_temp) / 2
            np.fill_diagonal(P_norm, 1/2)
            P_new[i] = P_norm
            
    
    alpha = softmax(alpha)
    P_int = graph_integration(np.array(P_new), alpha)
    
    return P_int



def to_torch_graph(W, y, train_index, val_index, test_index, X = None):
    
    edge_index, edge_weight = from_scipy_sparse_matrix(W)
    
    y = torch.tensor(y, dtype = torch.int64)
    train_mask = index_to_mask(torch.tensor(train_index), size = W.shape[0])
    val_mask = index_to_mask(torch.tensor(val_index), size = W.shape[0])
    test_mask = index_to_mask(torch.tensor(test_index), size = W.shape[0])
    
    if X is None:
        I = torch.tensor(np.eye(W.shape[0]), dtype = torch.float32)
        graph = Data(x=I, edge_index=edge_index, y=y)
    else:
        X = torch.tensor(X, dtype = torch.float32)
        graph = Data(x=X, edge_index=edge_index, y=y)
        
    graph.edge_weight = edge_weight.float()
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    
    return graph



def Training_History(train_history, val_history, model_name):
    
    plt.plot(train_history)
    plt.plot(val_history)
    plt.title('{}'.format(model_name), fontname = 'Cambria', fontsize = 15)
    plt.xlabel('Epoch', fontname = 'Cambria', fontsize = 12)
    plt.ylabel('Loss', fontname = 'Cambria', fontsize = 12)
    plt.legend(['Train', 'Valid'], loc='upper right', prop = font)
    plt.show()


def AUCList_Boarding(AUC_df_li, AUC_df_cv_li, rep, c):
    
    for AUC_df, AUC_df_cv in zip(AUC_df_li, AUC_df_cv_li):
        AUC_df.loc[rep, c] = AUC_df_cv.mean().values[0]


def AUCList_Averaging(AUC_df_li):
    
    for AUC_df in AUC_df_li:
        AUC_df.loc['Avg', :] = AUC_df.mean()
        AUC_df.loc['Std', :] = AUC_df.std()  


def AUCList_Save(AUC_df_li, save_directory, model_name):
    
    for num, AUC_df in enumerate(AUC_df_li):
        AUC_df.to_excel(save_directory + 'AUC_{}(graph{}, {}rep, {}%).xlsx'.format(model_name, num+1, NUM_REP, LABEL_RATE*100))



#%% Data Load

title = 'BRCA'

# figure_directory = 'C:/Users/YTH/Desktop/EXP_MGD/figure/' # Change Directory

directory = 'C:/Users/YTH/Desktop/EXP_MGD/data/' # Change Directory

with open(directory + "BRCA_mod.pickle","rb") as fr:
    data = pickle.load(fr)

W_total = data['W_total']

L_total = multiplex_Laplacian_csc(W_total)
    
label = data['label']
    
num_category = len(np.unique(label))
identity_mat = np.eye(num_category)
label = identity_mat[label]

num_graph = W_total.shape[0]
num_node = W_total[0].shape[0]
num_class = 2 # binary classification in model

#%%


# Learning Setting
NUM_EPOCH = 300
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
PATIENCE = 20
DELTA = 0
MODE = 'min'
VERBOSE = False

# GCN Setting
NUM_HIDDEN_sGCN = [8] * num_graph
NUM_HIDDEN_iGCN = 8
NUM_HIDDEN_GDCGCN = 8
NUM_HIDDEN_SNFGCN = 8
NUM_HIDDEN_SNFGCN_star = 8
NUM_HIDDEN_SNFGCN_P = 8
NUM_HIDDEN_SNFGCN_P_star = 8

# SNF Setting
ITERATION_SNF = 20
ITERATION_SNF_star = 20
ITERATION_SNF_P = 20
ITERATION_SNF_P_star = 20

# Experimental Setting
# LABEL_RATE = 0.2  ## Fixed
# LABEL_RATE = 0.1
# LABEL_RATE = 0.05
LABEL_RATE = 0.01
NUM_REP = 10

kfold = StratifiedKFold(n_splits = int(1/LABEL_RATE), shuffle = True)

class GCN(torch.nn.Module):
    def __init__(self, num_node, num_hidden, num_class):
        super().__init__()
        self.conv = GCNConv(num_node, num_hidden, add_self_loops=True)
        self.lin = torch.nn.Linear(num_hidden, num_class)

    def forward(self, graph):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_weight

        x = self.conv(x=x, edge_index=edge_index, edge_weight = edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.lin(x)

        return F.softmax(x, dim=1)


def train_GCN(graph, num_node, num_hidden, num_class, model_name):
    
    gcn = GCN(num_node, num_hidden, num_class).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
    es = EarlyStopping(patience = PATIENCE,
                       delta = DELTA,
                       mode = MODE,
                       verbose = VERBOSE)
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(NUM_EPOCH):
        gcn.train()
        
        optimizer.zero_grad()
        out = gcn(graph)
        loss = F.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
        train_loss_history.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        
        gcn.eval()
        with torch.no_grad():    
            val_loss = F.cross_entropy(out[graph.val_mask], graph.y[graph.val_mask])
            val_loss_history.append(val_loss.detach().cpu().numpy())
            es(val_loss.detach().cpu().numpy())
            if es.early_stop:
                break
        scheduler.step()
        
    Training_History(train_loss_history, val_loss_history, model_name)
    
    return gcn


def test_model(model, graph, y_real, test_index):
    
    model.eval()
    with torch.no_grad():
    
        y_pred = model(graph)[:, 1]
        y_pred = y_pred[test_index].detach().cpu().numpy()
    
    y_real = y_real.copy()
    fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label = 1)
    auc_score = auc(fpr, tpr)
    
    return auc_score


def test_ensmeble_model(model_li, graph_li, y_real, test_index):
    
    y_pred = np.zeros(len(test_index))
    for model, graph in zip(model_li, graph_li):
        model.eval()
        with torch.no_grad():
            
            y_pred_temp = model(graph)[:, 1]
            y_pred_temp = y_pred_temp[test_index].detach().cpu().numpy()
            y_pred = y_pred + y_pred_temp
    
    y_pred = y_pred / len(model_li)
    y_real = y_real.copy()
    fpr, tpr, thresholds = roc_curve(y_real, y_pred, pos_label = 1)
    auc_score = auc(fpr, tpr)
    
    return auc_score

#%%

# Comparison
AUC_sGCN = [pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1)) for i in range(num_graph)]
AUC_iGCN = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1))
AUC_eGCN = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1))
AUC_GDCGCN = [pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1)) for i in range(num_graph)]

# Proposed
AUC_SNFGCN = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1))
AUC_SNFGCN_star = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1))
AUC_SNFGCN_P = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1))
AUC_SNFGCN_P_star = pd.DataFrame(index = range(1, NUM_REP + 1), columns = range(1, label.shape[1] + 1))


for c in range(label.shape[1]):
    y = label.copy()[:, c]
    
    for rep in range(1, NUM_REP + 1):
        
        print('Repetition: {}  -----------------------------'.format(rep))
        
        
        AUC_cv_sGCN = [pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC']) for i in range(num_graph)]
        AUC_cv_iGCN = pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC'])
        AUC_cv_eGCN = pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC'])
        AUC_cv_GDCGCN = [pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC']) for i in range(num_graph)]
        AUC_cv_SNFGCN = pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC'])
        AUC_cv_SNFGCN_star = pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC'])
        AUC_cv_SNFGCN_P = pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC'])
        AUC_cv_SNFGCN_P_star = pd.DataFrame(index = list(range(1, int(1/LABEL_RATE) + 1)), columns = ['AUC'])
        
        
        for fold, (no_train_index, train_index) in enumerate(kfold.split(np.array(range(len(y.copy()))), y)):
            
            if fold >= 5:
                break
            
            torch.cuda.empty_cache()
            val_index, test_index = train_test_split(no_train_index, test_size = 0.8, stratify = y.copy()[no_train_index])
            
            y_real = y.copy()[test_index]
            
            # Label for Integration
            y_ssl = y.copy()
            y_ssl[y_ssl == 0] = -1
            y_ssl[no_train_index] = 0
            
            
            graph_li = []
            for k in range(num_graph):
                graph = to_torch_graph(W_total[k], y.copy(), train_index, val_index, test_index)
                graph_li.append(graph)
                
            
            # sGCN Start
            sGCN_li = []
            for (k, num_hidden) in zip(range(num_graph), NUM_HIDDEN_sGCN):
                model_GCN = train_GCN(graph_li[k].to(device), num_node, num_hidden, num_class,
                                      'sGCN for Graph {}'.format(k + 1))
                sGCN_li.append(model_GCN)
                auc_score = test_model(model_GCN, graph_li[k], y_real, test_index)
                AUC_cv_sGCN[k].loc[fold+1, 'AUC'] = auc_score
                
            
            
            # iGCN Start
            alpha = SGF(L_total, y_ssl, train_index)
            alpha = softmax(alpha)
            W_int = graph_integration(W_total, alpha)
            
            graph_iGCN = to_torch_graph(W_int, y.copy(), train_index, val_index, test_index)
            model_iGCN = train_GCN(graph_iGCN.to(device), num_node, NUM_HIDDEN_iGCN, num_class, 'iGCN')
            auc_score = test_model(model_iGCN, graph_iGCN, y_real, test_index)
            AUC_cv_iGCN.loc[fold+1, 'AUC'] = auc_score
            
            # eGCN Start
            AUC_cv_eGCN.loc[fold+1, 'AUC'] = test_ensmeble_model(sGCN_li, graph_li, y_real, test_index)
            
            # SNFGCN Start (Proposed 1)
            torch.cuda.empty_cache()
            
            H_total = []
            
            for k in range(num_graph):
                H = sGCN_li[k].conv(graph_li[k].x, graph_li[k].edge_index, graph_li[k].edge_weight).detach().cpu().numpy()
                H = MinMaxScaler().fit_transform(H)
                H_total.append(H)
            
            W_SNF = SNF(H_total, W_total, ITERATION_SNF)
            graph_SNF = to_torch_graph(csc_matrix(W_SNF), y.copy(), train_index, val_index, test_index)
            model_SNFGCN = train_GCN(graph_SNF.to(device), num_node, NUM_HIDDEN_SNFGCN, num_class, 'SNFGCN')
            
            auc_score = test_model(model_SNFGCN, graph_SNF, y_real, test_index)
            AUC_cv_SNFGCN.loc[fold+1, 'AUC'] = auc_score
            
            
            # SNFGCN_star Start (Proposed 2)
            
            W_SNF = SNF_star(H_total, W_total, y_ssl, train_index, ITERATION_SNF_star)
            graph_SNF = to_torch_graph(csc_matrix(W_SNF), y.copy(), train_index, val_index, test_index)
            model_SNFGCN = train_GCN(graph_SNF.to(device), num_node, NUM_HIDDEN_SNFGCN_star, num_class, 'SNFGCN_star')
            
            auc_score = test_model(model_SNFGCN, graph_SNF, y_real, test_index)
            AUC_cv_SNFGCN_star.loc[fold+1, 'AUC'] = auc_score
            
            
            # SNFGCN_P Start (Proposed 3)
            
            W_SNF = SNF_P(H_total, W_total, y_ssl, train_index, ITERATION_SNF_P)
            graph_SNF = to_torch_graph(csc_matrix(W_SNF), y.copy(), train_index, val_index, test_index)
            model_SNFGCN = train_GCN(graph_SNF.to(device), num_node, NUM_HIDDEN_SNFGCN_P, num_class, 'SNFGCN_P')
            
            auc_score = test_model(model_SNFGCN, graph_SNF, y_real, test_index)
            AUC_cv_SNFGCN_P.loc[fold+1, 'AUC'] = auc_score
            
            
            # SNFGCN_P_star Start (Proposed 4)
            
            W_SNF = SNF_P_star(H_total, W_total, y_ssl, train_index, ITERATION_SNF_P_star)
            graph_SNF = to_torch_graph(csc_matrix(W_SNF), y.copy(), train_index, val_index, test_index)
            model_SNFGCN = train_GCN(graph_SNF.to(device), num_node, NUM_HIDDEN_SNFGCN_P_star, num_class, 'SNFGCN_P_star')
            
            auc_score = test_model(model_SNFGCN, graph_SNF, y_real, test_index)
            AUC_cv_SNFGCN_P_star.loc[fold+1, 'AUC'] = auc_score
            
            
            
        AUCList_Boarding(AUC_sGCN, AUC_cv_sGCN, rep, c+1)
        # AUCList_Boarding(AUC_GDCGCN, AUC_cv_GDCGCN, rep, c+1)
        
        AUC_iGCN.loc[rep, c+1] = AUC_cv_iGCN.mean().values[0]
        AUC_eGCN.loc[rep, c+1] = AUC_cv_eGCN.mean().values[0]
        AUC_SNFGCN.loc[rep, c+1] = AUC_cv_SNFGCN.mean().values[0]
        AUC_SNFGCN_star.loc[rep, c+1] = AUC_cv_SNFGCN_star.mean().values[0]
        AUC_SNFGCN_P.loc[rep, c+1] = AUC_cv_SNFGCN_P.mean().values[0]
        AUC_SNFGCN_P_star.loc[rep, c+1] = AUC_cv_SNFGCN_P_star.mean().values[0]


AUCList_Averaging(AUC_sGCN)
# AUCList_Averaging(AUC_GDCGCN)

AUC_iGCN.loc['Avg', :] = AUC_iGCN.mean()
AUC_iGCN.loc['Std', :] = AUC_iGCN.std() 

AUC_eGCN.loc['Avg', :] = AUC_eGCN.mean()
AUC_eGCN.loc['Std', :] = AUC_eGCN.std() 

AUC_SNFGCN.loc['Avg', :] = AUC_SNFGCN.mean()
AUC_SNFGCN.loc['Std', :] = AUC_SNFGCN.std()     

AUC_SNFGCN_star.loc['Avg', :] = AUC_SNFGCN_star.mean()
AUC_SNFGCN_star.loc['Std', :] = AUC_SNFGCN_star.std() 

AUC_SNFGCN_P.loc['Avg', :] = AUC_SNFGCN_P.mean()
AUC_SNFGCN_P.loc['Std', :] = AUC_SNFGCN_P.std() 

AUC_SNFGCN_P_star.loc['Avg', :] = AUC_SNFGCN_P_star.mean()
AUC_SNFGCN_P_star.loc['Std', :] = AUC_SNFGCN_P_star.std() 



#%%

save_directory = 'C:/Users/YTH/Desktop/EXP_MGD/result/{}/'.format(title)

AUCList_Save(AUC_sGCN, save_directory, 'sGCN')
# AUCList_Save(AUC_GDCGCN, save_directory, 'GDCGCN')

AUC_iGCN.to_excel(save_directory + 'iGCN_AUC({}rep, {}%).xlsx'.format(NUM_REP, LABEL_RATE*100))
AUC_eGCN.to_excel(save_directory + 'eGCN_AUC({}rep, {}%).xlsx'.format(NUM_REP, LABEL_RATE*100))
AUC_SNFGCN.to_excel(save_directory + 'SNFGCN_AUC({}rep, {}%).xlsx'.format(NUM_REP, LABEL_RATE*100))
AUC_SNFGCN_star.to_excel(save_directory + 'SNFGCN_star_AUC({}rep, {}%).xlsx'.format(NUM_REP, LABEL_RATE*100))
AUC_SNFGCN_P.to_excel(save_directory + 'SNFGCN_P_AUC({}rep, {}%).xlsx'.format(NUM_REP, LABEL_RATE*100))
AUC_SNFGCN_P_star.to_excel(save_directory + 'SNFGCN_P_star_AUC({}rep, {}%).xlsx'.format(NUM_REP, LABEL_RATE*100))
