# ProSPr: Protein Structure Prediction
Wendy Billings, Jacob Stern, Bryce Hedelius, Todd Millecam, David Wingate, Dennis Della Corte   
Brigham Young University     

This repository contains an open-source protein distance prediction network, ProSPr, released under the MIT license.

The manuscript corresponding to this work is available here:
https://www.biorxiv.org/content/10.1101/2021.10.14.464472v1

The preprint associated with a PREVIOUS VERSION (https://github.com/dellacortelab/prospr/tree/prospr1) is available here: https://www.biorxiv.org/content/10.1101/830273v2  

All data required to reproduce the work are publicly acessible at https://files.physics.byu.edu/data/prospr/

*************************************
WARNING: The 2TB of data associated with the PREVIOUS PROSPR VERSION currently hosted on the ftp server ARE SUBJECT TO BE REMOVED SOON. If you would like future access to the files, please download a local copy.
*************************************

### Running ProSPr

After downloading the code, a conda environment with all required dependencies can be created by running    
```
conda env create -f dependencies/prospr-env.yml
```   

```
pip install numba
# OR
conda install numba
```
(I used cherry picking installation of dependencies with conda and used pip when module not found error occured. (yamule))

Once activated
```
# Make a prediction:
python3 prospr.py predict --a3m ./data/evaluate/T1034.a3m

# Use precomputed hhm (Mostly for Windows, because hhsuite is not available with conda.)
python prospr.py  predict --a3m example_files\T1034_default.a3m_5.a3m --hhm example_files\T1034_default.a3m_5.hhm -o example_files\result

# Or train a new network
python3 prospr.py train

```
For more information, run    
```
python3 prospr.py -h
```
to print the help text.   


### Docker (only limited tested)
Alternatively to conda, you can use Docker. To run the code in a Docker container, run the following after cloning this repository:
```
cd prospr
# Build the docker image
docker build -t prospr-dev dependencies
# Run a docker container interactively
docker run -it --gpus all --name myname-prospr-dev --rm -v $(pwd):/code prospr-dev
# Then, inside the docker container, make a prediction:
cd code
python3 prospr.py predict --a3m ./data/evaluate/T1034.a3m
# Or train a new network
python3 prospr.py train
```

Contact: dennis.dellacorte@byu.edu
