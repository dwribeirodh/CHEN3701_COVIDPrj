#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 23:41:36 2020

@author: danielribeiro
"""
import os
import fastaparser
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sweetsourcod.lempel_ziv import lempel_ziv_complexity
from entropy_module import _get_entropy_rate, _get_rand_binary_lz77, get_entropy_rate_lz77
from sklearn.metrics import r2_score
import pdb

#%%
def _get_entropy_rate(c, nsites, norm=1, alphabetsize=2, method='lz77'):
    """
    :param c: number of longest previous factors (lz77) or unique words (lz78)
    :param norm: normalization constant, usually the filesize per character of a random binary sequence of the same length
    :param method: lz77 or lz78
    :return: entropy rate h
    """
    if method == 'lz77':
        h = (c * np.log2(c) + 2 * c * np.log2(nsites / c)) / nsites #entropy per amino acid
        #h /= norm
    elif method == 'lz78':
        h = c * (np.log2(alphabetsize) + np.log2(c)) / nsites 
        #h /= norm
    else:
        raise NotImplementedError
    return h

def get_entropy_sequence_lz77(seq):
    """
    Parameters
    ----------
    x : the array being compressed
    extrapolate : condition for guessing bin_size. The default is True.
    navg : number of times to average binary entropy

    Returns
    -------
    h_bound (The upper limit for the entropy level)
    h_sum_log (The numerical value for the entropy)
    """
    # now with LZ77 we compute the number of longest previous factors c, and the entropy rate h
    # note that lz77 also outputs the sum of the logs of the factors which is approximately
    # equal to the compressed size of the file
    nsites = seq.size
    alphabetsize = 20
    h_bound_bin = 1
    h_sumlog_bin = nsites
    c, h_sumlog = lempel_ziv_complexity(seq.astype('int'), 'lz77')
    h_bound = _get_entropy_rate(c, nsites, norm=h_bound_bin, alphabetsize=alphabetsize, method='lz77')
    h_sumlog /= h_sumlog_bin
    return h_bound, h_sumlog, nsites, c

def letter2number(sequence):

    sequence = sequence.replace('?', '')
    
    seqlist = []
    for item in sequence:
        seqlist.append(item)
    
    
    str_seq = []
    number_seq = []
    aa_dict = {"G":1, "A":2, "S":3, "T":4, "C":5, "V":6,
               "L":7, "I":8, "M":9, "P":10, "F":11, "Y":12,
               "W":13, "D":14, "E":15, "N":16, "Q":17,
               "H":18, "K":19, "R":20}
    
    for letter in seqlist:
        if letter in aa_dict:
            num = aa_dict[letter]
            str_seq.append(num)
        else:
            pass


    str_seq = np.array(str_seq)
    number_seq = str_seq.astype(int)
    return number_seq  

def number2letter(seqspace):
    num_dict = {1:"G", 2:"A", 3:"S", 4:"T", 5:"C", 6:"V",
               7:"L", 8:"I", 9:"M", 10:"P", 11:"F", 12:"Y",
               13:"W", 14:"D", 15:"E", 16:"N", 17:"Q",
               18:"H", 19:"K", 20:"R"}
    
    num_seq = []
    for number in seqspace:
        if number in num_dict:
            letter = num_dict[number]
            num_seq.append(letter)
        else:
            pass
            
    return np.array(num_seq)

def get_dec_h(array, delta):
    
    D = np.array([])
    entropy = np.array([])
    assert delta < len(array)
    while delta < len(array):
        shuffle = np.random.choice(array, delta)
        _ , h = get_entropy_rate_lz77(shuffle)
        entropy = np.append(entropy, h)
        delta += 1
        D = np.append(D, delta)
        
    return entropy , D
            

def parse_directory(directory):
    fname = []
    for entry in os.scandir(directory):
        if entry.path.endswith('.fasta'):
            fname.append(entry.path)            
    return fname
            
def get_sequence(fname):
    
    with open(fname) as fasta_file:
            parser = fastaparser.Reader(fasta_file)
            store = []
            for seq in parser:
                # seq is a FastaSequence object
                sequence = seq.sequence_as_string()
                store.append(sequence)
    
    return sequence

def log(x, a, b):
    return a + b*np.log(x)

def exp(x, a, b):
    return a*np.exp(b*x)
    
#%%
if __name__ == "__main__":
    
    directory = "/Users/danielribeiro/Desktop/Classes/Fall2020/CHEN3701/COVID_Project/Fasta_Files/"
    
    print("Parsing directory...")
    fname = parse_directory(directory)
      
    print("calculating entropy...")
    vname = []
    seq_entropy = np.array([])
    L = np.array([])
    C = np.array([])
    seqspace = np.array([])
    cid = np.array([])
    xi = np.array([])
    for file in fname:
        sequence = get_sequence(file)
        filename = file.replace('/Users/danielribeiro/Desktop/Classes/Fall2020/CHEN3701/COVID_Project/Fasta_Files/','')
        vname.append(filename)
        print(filename)
        number_sequence = letter2number(sequence)
        number_sequence = np.array(number_sequence)
        assert len(np.unique(number_sequence)) > 0
        H, D = get_dec_h(number_sequence, 1)
        CID = H / np.amax(H)
        plt.title('{}'.format(filename))
        plt.plot(D, CID)
        plt.show()
        print(len(np.unique(CID)))
        cid = np.append(cid, CID)
        XI = np.argmax(CID)
        corrl = D[XI]
        xi = np.append(xi, corrl)
        print('mean = {}'.format(np.mean(xi)))
        print('std = {}'.format(np.std(xi)))
        seqspace = np.append(seqspace, number_sequence)
        h_bound, h_sumlog, l, c = get_entropy_sequence_lz77(number_sequence)
        seq_entropy = np.append(seq_entropy, h_sumlog)
        L = np.append(L, l)
        C = np.append(C, c)
        
#%%        
    
    
    plt.figure(figsize=(8,4))
    plt.hist(xi, 10, density = True)
    plt.xlabel(r'$\xi$', fontsize = 12)
    plt.ylabel('Relative frequency', fontsize = 12)
        
    assert len(np.unique(seqspace)) == 20
        

    data = pd.DataFrame()
    data['Name'] = vname
    data['L'] = L
    data['C'] = C
    data['H'] = seq_entropy
    
    data.to_excel('viral_data.xlsx')
    
    R0 = pd.read_excel('R0.xlsx')
    entropy = R0['H']
    entropy = np.array(entropy)
    r0 = R0['R0']
    r0 = np.array(r0)
    
    
    x = np.linspace(0.001,590, 1000)
    xres = np.linspace(0.001, 590, len(seq_entropy))
    
    ### Plot 1: Entropy vs Length ###
    params, res = sp.optimize.curve_fit(log, L, seq_entropy)
    r2 = r2_score(seq_entropy, log(xres,*params))
    print(r2)
    
    plt.figure(figsize=(8,4))
    plt.grid(True)
    plt.xlabel("Amino Acid Sites", fontsize=12)
    plt.ylabel("Entropy (bits)", fontsize=12)
    plt.ylim(0,6)
    plt.xscale('log')
    plt.xlim(5,1000)
    plt.scatter(L, seq_entropy, c = 'orange')
    plt.plot(x,log(x,*params), c = 'navy')
    plt.show()
    
    
        
    
    ### Plot 2 ###
    param2, res = sp.optimize.curve_fit(exp, entropy, r0)
    
    plt.figure(figsize=(8,4))
    plt.scatter(entropy, r0, c = 'orange')
    plt.plot(x, exp(x,*params), c = 'navy')
    plt.grid(True)
    plt.xlabel('Entropy (bits)', fontsize = 12)
    plt.ylabel('$R_0$', fontsize = 12)
    plt.xlim(0,4.7)
    plt.ylim(0,14)
    
    
    
    
    ### Plot 3: AA Histogram ###
    plt.figure(figsize=(8,4))
    x = np.linspace(0,20,100)
    seqspace_letter = number2letter(seqspace)
    bins = len(np.unique(seqspace_letter))
    plt.hist(seqspace_letter, bins, histtype = 'stepfilled', density = True)
    plt.ylabel('Relative frequency', fontsize = 12)