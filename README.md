# PSD_MM
Matrix Multiplicate Update for Computing PSD Factorizations

This is the code base for numerical experiments for the paper
"A Non-Commutative Extension of Lee-Seung's Algorithm for Positive Semidefinite Factorizations"

Authors: Yong Sheng Soh, Antonios Varvitsiotis


Instructions

To replicate the experiment on Distance Matrices, run DistanceMatrix.ipynb

To view the results for the decomposition of faces, run FacesDecomposition.ipynb .  The pre-computed factorizations are included in this zip file / repo.
The code for pre-processing the CBCL image database is at FacesPreprocessing.ipynb .  The original datasource is *not* included -- instead, please use the following link http://www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz
The code for computing the factorizations in all instances is at ComputeBlockFactorizations.ipynb .  Note that the process will take a couple of hours.  Again, the processed data source is not included.  Instead, please obtain the original source file, apply the previous pre-processing step to obtain the processed data image file.


If you need help or find any errors, please contact Yong Sheng Soh at matsys@nus.edu.sg .