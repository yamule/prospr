#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import os
import subprocess
import multiprocessing
import torch
from datetime import datetime
from argparse import RawTextHelpFormatter
import warnings
warnings.simplefilter("ignore")
from prospr.io import load_model, Sequence, save
from prospr.nn import *
from prospr.prediction import domain,pred_with_files
from prospr.psiblast import PsiblastManager
from prospr.hhblits import *


# In[2]:


#If there are incompatibilities in pytorch versions, following process might help.
#model = torch.load("nn/ProSPr_full.nn")
#torch.save(model.state_dict(), "nn/ProSPr_full_new.pth")
#model = prospr();
#model.load_state_dict(torch.load("nn/ProSPr_full_new.pth"));
#model.eval();
#torch.save(model,"nn/ProSPr_full_converted.nn");


# In[ ]:



allowed = '-ACDEFGHIKLMNPQRSTVWY'

desc_usg = '''
Customized version of prospr https://github.com/dellacortelab/prospr
LGPL License
'''
diststart = 2.0;
diststep = 20.0/62.0;
def prob_to_dist(prospr_res):
    global diststep;
    global diststart;
    sshape = prospr_res.shape;
    ret = np.zeros(shape=(sshape[1],sshape[2]))
    prob = np.zeros(shape=(sshape[1],sshape[2]))
    for ii in range(sshape[1]):
        for jj in range(sshape[2]):
            pp = prospr_res[1:,ii,jj]*prospr_res[1:,jj,ii]
            psum = np.sum(pp);
            if psum <= 0.0:
                continue;
            amax = np.argmax(pp,axis=0);
            ret[ii,jj] = amax*diststep+diststart+diststep/2.0;
            prob[ii,jj] = pp[amax]/psum;
            ret[jj,ii] = ret[ii,jj];
            prob[jj,ii] = prob[ii,jj]; 
    return (ret,prob);


def prob_to_angle(prospr_res):
    sshape = prospr_res.shape;
    ret = np.zeros(shape=(sshape[1]));
    prob = np.zeros(shape=(sshape[1]));
    vstep = 360.0/36;
    for ii in range(sshape[1]):
        amax = np.argmax(prospr_res[1:,ii],axis=0);
        ret[ii] = amax*vstep-180.0;
        prob[ii] = prospr_res[amax+1,ii];
    return (ret,prob);

def prob_to_angle(prospr_res):
    sshape = prospr_res.shape;
    ret = np.zeros(shape=(sshape[1]));
    prob = np.zeros(shape=(sshape[1]));
    vstep = 360.0/36;
    for ii in range(sshape[1]):
        amax = np.argmax(prospr_res[1:,ii],axis=0);
        ret[ii] = amax*vstep-180.0;
        prob[ii] = prospr_res[amax+1,ii];
    return (ret,prob);


def main(args):
    network = load_model(args.network)
    if len(args.gpu) > 0:
        network.to(args.gpu);
        network.set_device_name(args.gpu)
    
    (dist_prob, phi_prob, psi_prob) = pred_with_files(args.fasta,args.pssm,args.hhm,args.mat,args.tmppkl, network, args.stride)
    slen = dist_prob.shape[1];
    dist_res = prob_to_dist(dist_prob);
    phi_res = prob_to_angle(phi_prob);
    psi_res = prob_to_angle(psi_prob);
    
    distfile = args.outfile+".dist.res";
    distbinfile = args.outfile+".distbin.res";
    anglefile = args.outfile+".phi_psi.res";
    with open(distfile,"w",newline="\n") as fout:
        for ii in range(slen):
            for jj in range(slen):
                if(ii >= jj):
                    continue;
                fout.write("r1:\t"+str(ii)+"\tr2:\t"+str(jj)
                           +"\tdist:\t"+str(dist_res[0][ii,jj])+"\tscore:\t"+str(dist_res[1][ii,jj])+"\n");
    with open(distbinfile,"w",newline="\n") as fout:
        sshape = dist_prob.shape;
        for ii in range(sshape[1]):
            for jj in range(sshape[2]):
                if ii < jj:
                    pp = dist_prob[1:,ii,jj]*dist_prob[1:,jj,ii]
                    psum = np.sum(pp);
                    if psum > 0.0:
                        pp = pp/psum;
                    fout.write("r1:\t{}\tr2:\t{}\tvalues:\t".format(ii,jj));
                    for q in range(pp.shape[0]):
                        if q > 0:
                            fout.write(",");
                        fout.write(str(pp[q]));
                    fout.write("\n");
    
    with open(anglefile,"w",newline="\n") as fout:
        for ii in range(slen):
            '''
            fout.write("res_num:\t"+str(ii)+"\tphi:\t"+str(phi_res[0][ii])+"\tphi_prob\t"+str(phi_res[1][ii])
                      +"\tpsi:\t"+str(psi_res[0][ii])+"\tpsi_prob\t"+str(psi_res[1][ii])+"\n");
            '''
            fout.write("res_num:\t"+str(ii)+"\tcategory:\tphi\t");
            fout.write("values:\t"+",".join( [str(ss) for ss in list(phi_prob[1:,ii])]));
            fout.write("\n");
            
            fout.write("res_num:\t"+str(ii)+"\tcategory:\tpsi\t");
            fout.write("values:\t"+",".join( [str(ss) for ss in list(psi_prob[1:,ii])]));
            fout.write("\n");
            
            
            
    save_path = args.outfile
        
    print('\nAnalyze prediction results: https://github.com/dellacortelab/prospr\n')
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc_usg, usage=argparse.SUPPRESS, formatter_class=RawTextHelpFormatter)
    subparsers = parser.add_subparsers(help="""Please check:
     run -h """, dest='command')
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('-n','--network', help='.nn file provided by dellacortelab.', default='full', required =True)
    run_parser.add_argument('-s','--stride', help='stride over which crops of domain are predicted and averaged, integer 1-30.\nWARNING: Using a small stride may result in very long processing time! Suggested for quick prediction: 25', type=int, default=25)
    run_parser.add_argument('-f','--fasta', help='Plain FASTA file.', default='', required =True)
    run_parser.add_argument('-p','--pssm', help='Ascii pssm file created by psi-blast', default='', required =True)
    run_parser.add_argument('-m','--mat', help='Customized plmDCA.jl result. (I think the archtecture must be the same with the original Prospr input. Please check example_files/2E74_D.pdb_d0.fas.jackali.max.dcares.dat.mat .)', default='', required =True)
    run_parser.add_argument('-b','--hhm', help='.hhm file by hhblits.', required =True)
    run_parser.add_argument('-t','--tmppkl', help='(output) intermediate pkl file.', default="tmp."+str(os.getpid())+".pkl")
    run_parser.add_argument('-o','--outfile', help='result file',  default="tmp."+str(os.getpid())+".res", required =True)
    run_parser.add_argument('-g','--gpu', help='gpu device name',  default="")
    
    args = parser.parse_args()
    if args.command == 'run':
        main(args)
    else:
        raise Exception();

