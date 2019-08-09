Noise Flow - A normalizing flows model for image noise modeling and synthesis
===

This repository provides the codes for training and testing the Noise Flow model used for imgage noise modeling and 
synthesis as described in the paper:

**Noise Flow: Noise Modeling with Conditional Normalizing Flows**

It also provides code for training and testing a CNN-based image denoiser (DnCNN) using Noise Flow as a noise generator,
 with compariosn to other noise generation methods (i.e., AWGN and signal-dependent noise).
  
#Required libraries

Python (works with 3.6)

TensorFlow (works with 1.12.0)

TensorFlow Probability (tested with 0.5.0)

_Despite not tested, the code may work with library versions other than the specified._

#Required dataset

[Smartphone Image Denoising Dataset (SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/)

#Training/Testing/Sampling

Start by running `job_noise_flow.sh`

It contains a set of examples for training different models (as in the paper) and optionally perform testing and 
sampling concurrently.

##Important parameters

`--sidd_path`: path to the SIDD dataset

`--arch`: the architecture of the noise flow model

`--cam`: (optional) to use/sample data from a specific camera

`--iso`: (optional) to use/sample data from a specific ISO level

Refer to `job_noise_flow.sh` or `ArgParser.py` for details on the rest of parameters.
   
#Sampling only

To use the Noise Flow trained model for generating noise samples:

Start by running `sample_noise_flow.py`

#Application to image denoising with DnCNN

_To be added._

#Paper

**Noise Flow: Noise Modeling with Conditional Normalizing Flows**

#Contact

Abdelrahman Abdelhamed ([kamel@eecs.yorku.ca](mailto:kamel@eecs.yorku.ca))