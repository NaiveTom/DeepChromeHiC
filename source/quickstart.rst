DeepChromeHiC Quickstart
========================

The final trainning & testing model with Friendly Interface.

This software can be used on both Windows and Linux platforms.





Module Introduction
+++++++++++++++++++

This release includes multiple models: 

- ``onehot_cnn_one_branch``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer together, and then use CNN to process, and finally get the output

- ``onehot_cnn_two_branch``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, and then send the promoter and enhancer to different CNN networks, and then merge the CNN networks to finally get the output

- ``onehot_embedding_dense``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer, then use the embedding layer to embed the gene fragment, and then send it to multiple dense layers and output

- ``onehot_embedding_cnn_one_branch``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer together, then use the embedding layer for embedding and then use CNN to process, and finally get the output

- ``onehot_embedding_cnn_two_branch``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, and then send the promoter and enhancer to different embedding layers for embedding and then send to different CNN networks, and then merge the CNN networks to finally get the output

- ``onehot_dense``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer, and then send it to multiple dense layers and output

- ``onehot_resnet18``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer, and then send it to The simplified (due to GPU memory usage) and improved resnet18, finally get the output.

- ``onehot_resnet34``

	Use one-hot encoding to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer, and then send it to The simplified (due to GPU memory usage) and improved resnet34, finally get the output.

- ``embedding_cnn_one_branch``

	Use dna2vec embedding (you can customize the training embedding matrix) to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer together, and then use CNN to process, and finally get the output

- ``embedding_cnn_two_branch``

	Use dna2vec embedding (you can customize the training embedding matrix) to process the ACGT base fragments of the promoter and enhancer, and then send the promoter and enhancer to different CNN networks, and then merge the CNN networks to finally get the output

- ``embedding_dense``

	Use dna2vec embedding (you can customize the training embedding matrix) to process the ACGT base fragments of the promoter and enhancer, then merge the promoter and enhancer together, and then use multiple dense layers to process, and finally get the output





System Requirements
+++++++++++++++++++

- The ``minimum memory size`` is ``16GB``. If you are dealing with a large amount of data, you need more memory to prepare the data.
- ``Graphics card memory minimum`` is ``8GB``
- (Optimal) The graphics card contains ``cuda`` and ``copy`` functions
- ``No other programs`` can occupy graphics card resources while running





Help Page
+++++++++

.. code:: bash

	D:\DeepChromeHiC>python DeepChromeHiC.py -h

	************************************
	************************************

	    Welcome to use DeepChromeHiC

	************************************
	************************************

	begin time >>> Fri Jun 18 10:31:09 2021



	If it is the first time to use, please preprocess the data first.
	Use the command: python3 DeepChromeHiC.py -p true -n [gene name]
	For example: python3 DeepChromeHiC.py -p true -n AD2.po

	usage: DeepChromeHiC.py [-h] [-p PREPROCESSING] [-m MODEL] [-t TYPE] [-n NAME]

	optional arguments:
	  -h, --help            show this help message and exit
	  -p PREPROCESSING, --preprocessing PREPROCESSING
							Preprocess the data, if you enter [true] (case sensitive), then proceed, if no, pass this
							process. Note: This command only needs to be entered once.
	  -m MODEL, --model MODEL
							Enter the model name which your choose: [onehot_cnn_one_branch] / [onehot_cnn_two_branch] /
							[onehot_embedding_dense] / [onehot_embedding_cnn_one_branch] /
							[onehot_embedding_cnn_two_branch] / [onehot_dense] / [onehot_resnet18] / [onehot_resnet34] /
							[embedding_cnn_one_branch] / [embedding_cnn_two_branch] / [embedding_dense] (all use
							lowercase).
	  -t TYPE, --type TYPE  Please choose [train] / [test] (all use lowercase).
	  -n NAME, --name NAME  Enter the gene name of your choice (note: case sensitive).
  
You can open the help interface by running ``python3 DeepChromeHiC.py -h``, which introduces the role of each parameter. 
 
 



.. image:: img/div.png



  
  
How to train and test a new gene
++++++++++++++++++++++++++++++++





Download
--------

Download all codes and data structures from DeepChromeHiC, https://github.com/NaiveTom/DeepChromeHiC.





Copy
----

Create a new folder called ``your gene name`` in the ``data`` folder, for example, ``AD2.po``.

Then paste your genetic data into this folder. 

- ``seq.anchor1.pos.txt`` 

	Put in the enhancer positive fragments

- ``seq.anchor1.neg2.txt`` 

	Put in the enhancer negative fragments

- ``seq.anchor2.pos.txt`` 

	Put in the promoter positive fragments

- ``seq.anchor2.neg2.txt`` 

	Put in the promoter negative fragments





.. note::

	- A total of 5 types of data can be identified, ``A``, ``C``, ``G``, ``T``, and ``N``.
	- Each piece of genetic data should occupy ``one row``.
	- The length of each gene data is ``10K`` (10001)





Coding your gene fragment
-------------------------

Excuting an order:

.. code:: bash

	python3 DeepChromeHiC.py -p true -n AD2.po

Then it will automatically start pre-processing AD2.po.

This step will generate ``png`` files and ``npz`` files.





.. note::

	- This instruction should be used once, if the previous data already exists, the previous data will be overwritten.
	- This program needs a lot of memory to run, and the time will be relatively long, generally about 15 minutes - 6 hours. Please wait patiently (there is no progress bar in the middle of the coding process, and the program is not stuck).
	- This code will split and save the training set and test set according to 0.9 : 0.1. During the training process, the training set will be split again into training set and validation set according to 0.89 : 0.11. In other words, the final training set: validation set: test set = 0.8 : 0.1 : 0.1.
	- This step will produce a large number of fragmented files, but the volume has been compressed by Huffman encoding, so it is recommended to use ``SSD`` for processing, which will significantly speed up the processing.





Run!
----

1. Open the ``cmd`` command line, and then enter your drive letter, such as 

	.. code:: bash

		D:
		
	Then cmd will enter the D drive: ``D:\``

2. Use the ``cd`` command to enter your project path, such as 

	.. code:: bash

		cd D:\DeepChromeHiC

3. Use the following command to train your data

	.. code:: bash
		
		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t train -n AD2.po
	
	For the meaning of the command statement, please refer to the Section: ``Help page``

4. Use the following command to test your data 

	.. code:: bash

		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n AD2.po
		
	For the meaning of the command statement, please refer to the Section: ``Help page``

5. The test and training results will be under the gene corresponding to the ``result`` folder. The model weights will be saved under the genes corresponding to the ``h5_weights`` folder.





Results
-------

- The graphical results are saved in the ``result`` folder.
- The auc_roc results of the test are saved in the ``log.txt`` for the convenience of subsequent data processing.





.. image:: img/div.png