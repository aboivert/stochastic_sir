#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is a modifaction of the original version of GEMFSim, with some added features (lockdown, curfew, R0)
Find more information about this version here : 
Find the original code here :
"""

# In[] - Imports
import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import random
import numpy.random as rand
from scipy.sparse import *
from scipy import *
from scipy.sparse import coo_matrix, bmat
import itertools
import time
import datetime
import os

# In[] - get adjacent nodes (not necessary neighbors)
def NeighborhoodData ( N , L1 , L2, W):
    junk = np.sort ( L1 ) 
    index = np.argsort( L1 )
    NeighVec = L2[index]
    NeighWeight = W[index]
    l = len( junk );     d = np.zeros ( N , dtype=int32) 
    I1 = np.zeros ( N , dtype=int32) ;
    I1 = -np.ones ( N , dtype=int32) 
    i = 0 
    while i+1 < l:  
        node = junk[i]
        I1[node] = i 
        while junk[i + 1] == junk[i]:
            d[node] = d[node] + 1 ;
            i += 1
            if i+1 == l:
                break
        i += 1
    if i+1 == l:
        node = junk[i]; I1[node] = i; d[node] = 0 
    I2 = I1 + d 
    Temp1 = np.subtract(I2,I1)
    Temp2 = [int(I1[i]!=0) for i in range(len(I1)) ]
    return NeighVec, I1, I2, d, NeighWeight   

# In[] - define initial condition
def Initial_Cond_Gen(N, J, NJ, x0):
    if sum(NJ) > N:
        return 'Oops! Initial infection is more than the total population'
    else:
        temp = np.random.permutation(N); nj=temp[0:sum(NJ)]
        for i in range(len(nj)):
            x0[nj[i]] = J
    return x0

# In[] - to draw a sample using a probability distribution
def rnd_draw(p):
    a = [0]
    a = np.append(a, np.cumsum(p[0:-1]))/np.sum(p)
    b = cumsum(p)/np.sum(p)
    toss = rand()
    k = np.intersect1d(np.nonzero(a<toss)[0], np.nonzero(b>=toss)[0])
    return k

# In[] - gives informations about graph (only for non-directed graphs)
def MyNet(G, weight=None):
    G_adj = nx.to_scipy_sparse_matrix(G, weight=weight)
    cx = G_adj.tocoo()  
    L2 = cx.row
    L1 = cx.col
    W = cx.data
    N = G.number_of_nodes()
    NeighVec, I1, I2, d, NeighWeight = NeighborhoodData ( N , L1 , L2, W)
    Net = [NeighVec, I1, I2, d, NeighWeight]#ver2
    return Net

def NetCmbn(NetSet):    
    Neigh = []; I1 = []; I2 = []; d = []; adj = []; NeighW = []
    for l in range(len(NetSet)):
        Neigh.append(NetSet[l][0]) 
        I1.append(NetSet[l][1])    
        I2.append(NetSet[l][2])
        d.append(NetSet[l][3])
        NeighW.append(NetSet[l][4])        
    Net = [Neigh,I1,I2,d, NeighW]
    return Net

# In[] - state counting
def Post_Population(x0, M, N, ts, i_index, j_index):
    X0 = np.zeros((M,N))
    for i in range(N):
        X0[int(x0[i])][i] = 1
    T = [0]
    T.extend(np.cumsum(ts))
    StateCount = np.zeros((M,len(ts)+1))
    StateCount[:,0] = X0.sum(axis=1)
    DX = np.zeros(M); DX[i_index[0]] = -1; DX[j_index[0]] = 1
    StateCount[:,1] = StateCount[:,0]+DX
    for k in range(len(ts)):
        DX = np.zeros(M); DX[i_index[k]] = -1; DX[j_index[k]] = 1
        StateCount[:,k+1] = StateCount[:,k] + DX
    return T, StateCount

# In[] - deterministic count
def MonteCarlo(graph, Net, Para, StopCond, Init_inf, M, step, nsim, N, Conf,Deconf,seuil_confinement,seuil_deconfinement,pos,curfew,x_init = None ):
    t_interval = np.arange(0,StopCond[1], step)    
    tsize = int(StopCond[1]/float(step))
    t_interval = np.linspace(0, StopCond[1], num=tsize)
    f = np.zeros(( M, tsize ))
    total_conf=0  
    total_deconf=0
    t_f_conf=0
    cpt_conf=0
    effective_n_sim=0
    R0f=np.zeros((tsize))
    for n in range(nsim): 
        x0 = Initial_Cond_Gen(N, Para[1][0], Init_inf, x0 = np.zeros(N, dtype = int32))
        [ts, n_index, i_index, j_index,R0,nb_conf,nb_deconf] = GEMF_SIM(graph, Para, Net, x0, StopCond, N,Conf,Deconf,seuil_confinement,seuil_deconfinement,pos,curfew)
        if curfew :
            try:
                [T, StateCount] = Post_Population(x0, M, N, ts, i_index, j_index,eve_conf)
                effective_n_sim+=1
                k=0
                R0int=np.zeros((tsize))
                y=np.zeros((M,tsize))
                NewT = T.extend([1000])
                for t in t_interval:
                    ind, tr = np.histogram(t,bins = T)
                    index = np.nonzero(ind)[0][0]
                    y[:,k] = StateCount[:, index]/N 
                    R0int[k]=R0[index]
                    k+=1
                R0f+=R0int   
                f += y;
                print("Simulation",n+1,"achevée")
            except:
                print("Problème simulation")
            if effective_n_sim==100:
                break
        else :
            [T, StateCount] = Post_Population(x0, M, N, ts, i_index, j_index)
            k=0
            R0int=np.zeros((tsize))
            y=np.zeros((M,tsize))
            NewT = T.extend([1000])
            for t in t_interval:
                ind, tr = np.histogram(t,bins = T)
                index = np.nonzero(ind)[0][0]
                y[:,k] = StateCount[:, index]/N
                R0int[k]=R0[index]
                k+=1
            R0f+=R0int   
            f += y;
            total_conf+=nb_conf
            total_deconf+=nb_deconf
            print("Simulation",n+1,"achevée")  
            effective_n_sim=100
#     fig = plt.figure(1000)
#     comp = ['S', 'I', 'R']
#     colors = ['olivedrab', 'tomato', 'gray']
#     col = dict(zip(comp, colors))
#     model = [x0, n_index, i_index, j_index]
#     pos=nx.get_node_attributes(graph, 'pos')
#     anim = animate_discrete_property_over_graph(graph, model, len(ts)-1, fig, n_index,i_index, j_index, comp, 'state',
#                                             col, pos = pos, Node_radius = .01)
#     plt.show()
#     anim.save('myTest11111.gif')
    return t_interval, f/effective_n_sim ,R0f/effective_n_sim,total_conf/effective_n_sim,total_deconf/effective_n_sim

# In[] - Gillespie implementation
def GEMF_SIM(graph,Para, Net, x0, StopCond, N, Conf, Deconf, seuil_confinement,seuil_deconfinement,pos,curfew,Directed = False):
    M = Para[0]; q = Para[1]; L = Para[2]; A_d = Para[3]; A_b = Para[4]
    Neigh = Net[0]; I1 = Net[1]; I2 = Net[2]; NeighW = Net[4]
    nb_conf=0
    nb_deconf=0
    n_index = []; j_index = []; i_index = []
    confinement=False
    curfew_index=1
    noeuds=np.array(graph.nodes())
    neigh=[]
    plt.show()
    for n in noeuds:
        neigh.append(list(graph.neighbors(n)))
    bil = np.zeros((M,L))
    for l in range(L):
        bil[:,l] = A_b[l].sum(axis=1)
    bi = np.zeros((M,M,L))
    for i in range(M):
        for l in range(L):
            bi[i, :, l] = A_b[l][i,:]
    di = A_d.sum(axis=1) 
    X = x0.astype(int32)
    Nq = np.zeros((L,N))
    for n in range(N):
        for l in range(L):
            Nln = Neigh[l][I1[l][n]:I2[l][n]+1]
            Nq[l][n] = sum((X[Nln]==q[l])*NeighW[l][I1[l][n]:I2[l][n]+1]        )     
    Rn = np.zeros(N)
    for n in range(N):
        Rn[n] = di[X[n]] + np.dot(bil[X[n],:],Nq[:,n])
    R = sum(Rn)
    EventNum = StopCond[1]; RunTime= StopCond[1]    
    ts = []
    R0=[]
    compteur = 0
    noeuds_pos=[]
    for i in range(len(X)):
       if X[i] == 1:
           compteur+=1 
           noeuds_pos=np.concatenate((noeuds_pos,neigh[i]))
    noeuds_pos=np.sort(noeuds_pos)
    liste_noeuds=np.unique(noeuds_pos)
    for i in liste_noeuds:
        if X[int(i)]==2 or X[int(i)]==1:
            noeuds_pos=np.setdiff1d(noeuds_pos,i)
    if compteur==0:
        R0.append(0)
    else:
        R0.append(len(noeuds_pos)/compteur)
    s=-1; Tf=0 
    while Tf < RunTime:
        s +=1
        ts.append(-log( rand() )/R)        
        ns = rnd_draw(Rn)
        iss = X[ns]
        js = rnd_draw( np.ravel(A_d[iss,:].T  + np.dot(bi[iss],Nq[:,ns]) ))
        n_index.extend(ns)
        j_index.extend(js)
        i_index.extend(iss)
        X[ns] = js
        R -= Rn[ns]
        Rn[ns] = di[js] + np.dot(bil[js,:] , Nq[:,ns])
        R += Rn[ns]       
        infl = (q == js).nonzero()[0]     
        for l in infl:          
            Nln = Neigh[int(l)][int(I1[l][ns]):int(I2[l][ns])+1] 
            IncreasEff = NeighW[int(l)][int(I1[l][ns]):int(I2[l][ns])+1]
            Nq[l][Nln] += IncreasEff 
            k = 0
            for n in Nln:
                Rn[n] += bil[X[n],l]*IncreasEff[k]
                R += bil[X[n],l]*IncreasEff[k]
                k +=1        
        infl2 = (q == iss).nonzero()[0]
        for l in infl2: 
            Nln = Neigh[int(l)][int(I1[l][ns]):int(I2[l][ns])+1] 
            reducEff = NeighW[int(l)][int(I1[l][ns]):int(I2[l][ns])+1]
            Nq[l][Nln] -= reducEff 
            k = 0
            for n in Nln:  
                Rn[n] -= bil[X[n],l]*reducEff[k]         
                R -= bil[X[n],l]*reducEff[k]
                k += 1
        Tf += ts[s] 
        compteur = 0
        noeuds_pos=[]
        for i in range(len(X)):
            if X[i] == 1:
                compteur+=1 
                noeuds_pos=np.concatenate((noeuds_pos,neigh[i]))
        noeuds_pos=np.sort(noeuds_pos)
        liste_noeuds=np.unique(noeuds_pos)
        for i in liste_noeuds:
            if X[int(i)]==2 or X[int(i)]==1:
                noeuds_pos=np.setdiff1d(noeuds_pos,i)
        if compteur==0:
            R0.append(0)
        else:
            R0.append(len(noeuds_pos)/compteur)
        if (Conf == True and confinement==False and compteur/len(X) >seuil_confinement):
            graphe=nx.random_geometric_graph(N,0.035,pos=pos)
            Net=NetCmbn([MyNet(graphe)])
            print("Mise en place du confinement, Tf = ", Tf)
            Neigh = Net[0]; I1 = Net[1]; I2 = Net[2]; NeighW = Net[4]
            confinement = True
            nb_conf+=1
            Nq = np.zeros((L,N))
            for n in range(N):
                for l in range(L): 
                    Nln = Neigh[l][I1[l][n]:I2[l][n]+1]
                    Nq[l][n] = sum((X[Nln]==q[l])*NeighW[l][I1[l][n]:I2[l][n]+1]) 
            Rn = np.zeros(N)
            for n in range(N):
                Rn[n] = di[X[n]] + np.dot(bil[X[n],:],Nq[:,n])
            R = sum(Rn)
        if (Deconf == True and confinement==True and compteur/len(X) <seuil_deconfinement):
            graphe=nx.random_geometric_graph(N,0.045,pos=pos)
            Net=NetCmbn([MyNet(graphe)])
            Neigh = Net[0]; I1 = Net[1]; I2 = Net[2]; NeighW = Net[4]
            confinement = False
            nb_deconf+=1
            print("Déconfinement, Tf = ", Tf)
            Nq = np.zeros((L,N))
            for n in range(N):
                for l in range(L): 
                    Nln = Neigh[l][I1[l][n]:I2[l][n]+1]
                    Nq[l][n] = sum((X[Nln]==q[l])*NeighW[l][I1[l][n]:I2[l][n]+1]) 
            Rn = np.zeros(N)
            for n in range(N):
                Rn[n] = di[X[n]] + np.dot(bil[X[n],:],Nq[:,n])
            R = sum(Rn)  
        if curfew and Tf>curfew_index-1/3:
            print("Début du couvre-feu, Tf = ", Tf)
            graphe=nx.random_geometric_graph(N,0.032,pos=pos)
            Net=NetCmbn([MyNet(graphe)])
            Neigh = Net[0]; I1 = Net[1]; I2 = Net[2]; NeighW = Net[4]
            Nq = np.zeros((L,N))
            for n in range(N):
                for l in range(L): 
                    Nln = Neigh[l][I1[l][n]:I2[l][n]+1]
                    Nq[l][n] = sum((X[Nln]==q[l])*NeighW[l][I1[l][n]:I2[l][n]+1]) 
            Rn = np.zeros(N)
            for n in range(N):
                Rn[n] = di[X[n]] + np.dot(bil[X[n],:],Nq[:,n])
            while Tf<curfew_index and Tf>curfew_index-1/3:
                s +=1
                R = sum(Rn)
                ts.append(-log( rand() )/R)
                ns = rnd_draw(Rn)  
                iss = X[ns] 
                js = rnd_draw( np.ravel(A_d[iss,:].T  + np.dot(bi[iss],Nq[:,ns]) )) 
                n_index.extend(ns)
                j_index.extend(js)
                i_index.extend(iss)
                X[ns] = js 
                R -= Rn[ns] 
                Rn[ns] = di[js] + np.dot(bil[js,:] , Nq[:,ns])
                R += Rn[ns] 
                infl = (q == js).nonzero()[0] 
                for l in infl:        
                    Nln = Neigh[int(l)][int(I1[l][ns]):int(I2[l][ns])+1] 
                    IncreasEff = NeighW[int(l)][int(I1[l][ns]):int(I2[l][ns])+1]
                    Nq[l][Nln] += IncreasEff 
                    k = 0
                    for n in Nln: 
                        Rn[n] += bil[X[n],l]*IncreasEff[k]
                        R += bil[X[n],l]*IncreasEff[k]
                        k +=1        
                infl2 = (q == iss).nonzero()[0] 
                for l in infl2: 
                    Nln = Neigh[int(l)][int(I1[l][ns]):int(I2[l][ns])+1] 
                    reducEff = NeighW[int(l)][int(I1[l][ns]):int(I2[l][ns])+1]
                    Nq[l][Nln] -= reducEff 
                    k = 0
                    for n in Nln:   
                        Rn[n] -= bil[X[n],l]*reducEff[k]         
                        R -= bil[X[n],l]*reducEff[k]
                        k += 1
                compteur = 0
                noeuds_pos=[]
                for i in range(len(X)):
                    if X[i] == 1:
                        compteur+=1 
                        noeuds_pos=np.concatenate((noeuds_pos,neigh[i]))
                noeuds_pos=np.sort(noeuds_pos)
                liste_noeuds=np.unique(noeuds_pos)
                for i in liste_noeuds:
                    if X[int(i)]==2 or X[int(i)]==1:
                        noeuds_pos=np.setdiff1d(noeuds_pos,i)
                if compteur==0:
                    R0.append(0)
                else:
                    R0.append(len(noeuds_pos)/compteur)
                Tf += ts[s]
            graphe=nx.random_geometric_graph(N,0.051,pos=pos)
            Net=NetCmbn([MyNet(graphe)])
            Neigh = Net[0]; I1 = Net[1]; I2 = Net[2]; NeighW = Net[4]
            Nq = np.zeros((L,N))
            for n in range(N):
                for l in range(L): 
                    Nln = Neigh[l][I1[l][n]:I2[l][n]+1]
                    Nq[l][n] = sum((X[Nln]==q[l])*NeighW[l][I1[l][n]:I2[l][n]+1]) 
            Rn = np.zeros(N)
            for n in range(N):
                Rn[n] = di[X[n]] + np.dot(bil[X[n],:],Nq[:,n])
            R = sum(Rn)  
            curfew_index+=1
            print("Fin du couvre-feu, Tf = ", Tf)
        if R < 1e-6:
            break
    return ts, n_index, i_index, j_index, R0, nb_conf, nb_deconf

# In[] - various models
def Para_SIR(delta, beta):
    M = 3; q = np.array([1]); L = len(q);
    A_d = np.zeros((M,M));   A_d[1][2] = delta
    A_b = []
    for l in range(L):
        A_b.append(np.zeros((M,M)))
    A_b[0][0][1] = beta
    Para=[M,q,L,A_d,A_b]
    return Para

def Para_SIS(delta,beta):
    M = 2; q = np.array([1]); L = len(q);
    A_d = np.zeros((M,M)); A_d[1][0] = delta
    A_b = []
    for l in range(L):
        A_b.append(np.zeros((M,M)))
    A_b[0][0][1] = beta 
    Para=[M,q,L,A_d,A_b]
    return Para

# In[] - animation gif
from matplotlib import animation
def animate_discrete_property_over_graph( g, model, steps, fig, n_index,i_index, j_index, comp, property = None,
                                         color_mapping = None, pos = None, Node_radius = None, **kwords ):
    x0 = model[0]; n_index = model[1]; i_index = model[2]; j_index = model[3]
    ax = fig.gca()
    pos
    ax.grid(False)                
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])
    nx.draw_networkx_edges(g, pos)
    if Node_radius == None:
        Node_radius = .02
    nodeMarkers = []
    for v in g.nodes_iter():
        circ = plt.Circle(pos[v], radius = Node_radius, zorder = 2)  
        ax.add_patch(circ)
        nodeMarkers.append({ 'node_key': v, 'marker': circ })
    def colour_nodes():
        for nm in nodeMarkers:
            v = nm['node_key']
            state = g.node[v][property]
            c = color_mapping[state]
            marker = nm['marker']
            marker.set(color = c)
    def init_state():
        """Initialise all node in the graph to be susceptible."""
        for i in g.node.keys():
            g.node[i]['state'] = comp[int(x0[i])]      
        colour_nodes()
    def frame(i):
        changing_node = n_index[i]
        new_comp = j_index[i]
        g.node[changing_node]['state'] = comp[new_comp]
        colour_nodes()
    return animation.FuncAnimation(fig, frame, init_func = init_state, frames = steps, **kwords)
