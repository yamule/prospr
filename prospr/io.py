from prospr.common import findHashTag, probability, findRows
from prospr import pconf
from prospr.nn import *
import numpy as np
import torch
import pickle as pkl
from scipy.io import loadmat
import re

np.set_printoptions(threshold=np.inf)

def save(arr,fileName):
    fileObject = open(fileName, 'wb')
    pkl.dump(arr, fileObject)
    fileObject.close()

def load(fileName):
    fileObject2 = open(fileName, 'rb')
    modelInput = pkl.load(fileObject2)
    fileObject2.close()
    return modelInput

def load_model(fname):
    return torch.load(fname)

class Sequence(object):
    def __init__(s, name, **kwargs):
        """Accepts name as input and expects a corresponding .pssm, hhm, mat, and fastfa txt file (dashes represent gaps), and outputs a pkl file that can be used by ProSPr to predict contacts"""
        s.name = name 
        s.base = pconf.basedir + s.name + "/" + s.name 
        s.data = dict()
        s.outfile = s.base + ".pkl" 
        
        
    @staticmethod
    def build_with_files(seq_file,pssm_file,hhm_file,mat_file,out_file):
        #inspired by the work by Ag_smith https://qiita.com/Ag_smith/items/849abf7ed95e0d57d1b0
        
        s = Sequence("dummy")
        
        filename = pssm_file 
        pssm_validlines=[];
        with open(filename) as f:
            data = f.readlines()
            for ll in data:
                if(re.search("^[\s]*[0-9]+[\s]+[A-Za-z][\s]",ll)):
                    pssm_validlines.append(ll);
            
            
            
            
        NUM_ROWS = len(pssm_validlines);
        NUM_COL = 20
        matrix = np.zeros((NUM_ROWS,NUM_COL))
        for x in range(NUM_ROWS):
            line = pssm_validlines[x].split()[2:22]
            for i, element in enumerate(line):
                matrix[x,i] = element
        s.data['PSSM'] = matrix
        
        
        filename = hhm_file
        with open(filename) as f:
            data = f.readlines()
        NUM_COL = 30
        NUM_ROW = findRows(filename)
    
        pssm = np.zeros((NUM_ROW, NUM_COL))
    
        line_counter = 0
    
        start = findHashTag(data)-1
    
        for x in range (0, NUM_ROW * 3):
            if x % 3 == 0:
                line = data[x + start].split()[2:-1]
                for i, element in enumerate(line):
                    prop = probability(element)
                    pssm[line_counter,i] = prop
            elif x % 3 == 1:
                line = data[x+start].split()
                for i, element in enumerate(line):
                    prop = probability(element)
                    pssm[line_counter, i+20] = prop
                line_counter += 1
        s.data['HH'] = pssm
        
        
        # probably need to pull this filename in from the args
        filename = seq_file;
        
        seqs=[];
        with open(filename) as f:
            for ll in f:
                if re.search(">",ll):
                    if len(seqs) > 0:
                        raise Exception("Fasta file format error. Does it have multiple sequence?");
                    continue;
                ll = re.sub("[\s]","",ll);
                seqs.append(ll);
                
            s.data['seq'] = "".join(seqs);
        
        filename = mat_file
        try:
            potts = loadmat(filename)
            s.data['J'] = potts['J'].astype('float16')
            s.data['h'] = potts['h']
            s.data['frobenius_norm'] = potts['frobenius_norm']
            s.data['score'] = potts['score']

        except:
            import h5py
            potts = h5py.File(filename, 'r');
            s.data['J'] = np.array(potts['J'].value).astype(np.float16)
            s.data['h'] = np.array(potts['h'].value)
            s.data['frobenius_norm'] = np.array(potts['frobenius_norm'].value)
            s.data['score'] = np.array(potts['score'].value)
            
        pshape=s.data['J'].shape
        #print(pshape);
        if pshape[3] != 21 and pshape[3] != 22:
            raise Exception("The shape of J tensor is "+str(pshape)+" Does your sequence have all 20 amino acid types? If not, it cannot be processed...");
        if pshape[3] == 21:
            pdat=np.zeros((pshape[0],pshape[1],22,22),dtype=np.float16);
            #Prospr は第一番目にギャップを使うので
            #pdat[:,:,1:22,1:22] = s.data['J'];#悪い
            
            pdat[:,:,1:21,1:21] = s.data['J'][:,:,0:20,0:20];
            pdat[:,:,0:1,0:1] = s.data['J'][:,:,20:21,20:21];
            pdat[:,:,1:21,0:1] = s.data['J'][:,:,0:20,20:21];
            pdat[:,:,0:1,1:21] = s.data['J'][:,:,20:21,0:20];
            
            #pdat[:,:,1:22,1:22] = s.data['J'][:,:,0:21,0:21];#ものすごく悪い
            #pdat[:,:,0:1,21:22] = s.data['J'][:,:,20:21,20:21];
            #pdat[:,:,21:22,0:1] = s.data['J'][:,:,20:21,20:21];
            #pdat[:,:,0:1,0:1] = s.data['J'][:,:,20:21,20:21];
            #pdat[:,:,1:21,0:1] = s.data['J'][:,:,0:20,20:21];
            #pdat[:,:,0:1,1:21] = s.data['J'][:,:,20:21,0:20];
            
            #pdat[:,:,1:21,1:21] = s.data['J'][:,:,0:20,0:20];
            
            s.data['J'] = pdat;
            
            
            pshape=s.data['h'].shape
            #print(pshape);
            if pshape[1] == 21:
                pdat=np.zeros((pshape[0],22),s.data['h'].dtype);
                #pdat[:,1:22] = s.data['h'];
                
                pdat[:,1:21] = s.data['h'][:,0:20];
                pdat[:,0:1] = s.data['h'][:,20:21];
                
                #pdat[:,1:22] = s.data['h'][:,0:21];
                #pdat[:,0:1] = s.data['h'][:,20:21];
                
                #pdat[:,1:21] = s.data['h'][:,0:20];
                
                
                s.data['h'] = pdat;
            
            
        
        
        save(s.data, out_file)
        
    def build(s, args):
        """Builds the pkl file for the protein sequence and saves it.  `args` is an argparse object and it will look, specifically, for a boolean args.potts"""
        s.seq_name()
        s.pssm()
        s.hh()
        s.potts()
        filename = s.base + ".pkl"
        save(s.data, filename)

    def pssm(s): 
        filename = s.base + ".pssm" 
        with open(filename) as f:
            data = f.readlines()
        count = len(data)
        NUM_ROWS = count - 9
        NUM_COL = 20
        matrix = np.zeros((NUM_ROWS,NUM_COL))
        for x in range(NUM_ROWS):
            line = data[x + 3].split()[2:22]
            for i, element in enumerate(line):
                matrix[x,i] = element
        s.data['PSSM'] = matrix

    def hh(s): 
        filename = s.base + ".hhm"
        with open(filename) as f:
            data = f.readlines()
        NUM_COL = 30
        NUM_ROW = findRows(filename)
    
        pssm = np.zeros((NUM_ROW, NUM_COL))
    
        line_counter = 0
    
        start = findHashTag(data)-1
    
        for x in range (0, NUM_ROW * 3):
            if x % 3 == 0:
                line = data[x + start].split()[2:-1]
                for i, element in enumerate(line):
                    prop = probability(element)
                    pssm[line_counter,i] = prop
            elif x % 3 == 1:
                line = data[x+start].split()
                for i, element in enumerate(line):
                    prop = probability(element)
                    pssm[line_counter, i+20] = prop
                line_counter += 1
        s.data['HH'] = pssm
    
         
    def seq_name(s):
        # probably need to pull this filename in from the args
        filename = s.base + ".fasta"
        with open(filename) as f:
            s.data['seq'] = f.readlines()[1]

    def potts(s):
        filename = s.base + ".mat"
        potts = loadmat(filename)
        s.data['J'] = potts['J'].astype('float16')
        s.data['h'] = potts['h']
        s.data['frobenius_norm'] = potts['frobenius_norm']
        s.data['score'] = potts['score']

    
