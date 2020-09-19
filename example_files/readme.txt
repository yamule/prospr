1L9Y is an example which is much more appropriate to see the predictor's performance.
However, as the file for Potts model is really large that you have to create by yourself.
plmDCA.jl can be used. Please follw the instruction in ../potts_plmDCA.jl/howto.txt.
& Type
julia ../potts_PlmDCA.jl/run_plmDCA.jl ./1L9Y/1L9Y_A.pdb_d0.msa 1L9Y/1L9Y_A.pdb_d0.dcares

python ../run.py run -n ../nn/ProSPr_full_converted.nn -p 1L9Y/1L9Y_A.pdb_d0.pssm.ascii  -b 1L9Y/1L9Y_A.pdb_d0.hhm -m 1L9Y/1L9Y_A.pdb_d0.dcares.mat -g "cuda:0" -f 1L9Y/1L9Y_A.pdb_d0.fas  -o 1L9Y/1L9Y_A.pdb_d0.prosprres 

yamule