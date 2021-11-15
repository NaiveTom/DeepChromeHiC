DeepChromeHiC.py
================

DeepChromeHiC.py is the root file of this project, responsible for calling all subroutines.

DeepChromeHiC is an arguments parser.





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
  
  
  
  

Command Example
+++++++++++++++



- ``python3 DeepChromeHiC.py -h``

	- ``-h``
	
	This parameter is used to request ``Help Page``

Entering this parameter will output the ``Help Page`` (See the previous unit for details).



- ``python3 DeepChromeHiC.py -p true -n AD2.po``

	- ``-p true``
	
	This parameter is used to prepare data, if it is ``true`` , png files and npz files are generated.
	This is not a required parameter, so if you are training, you do not need to write this parameter.
	
	- ``-n AD2.po``
	
	This parameter is used to inform the system of the gene name to be operated.
	The name of the gene to be operated here is ``AD2.po``

Using this command will prepare all the ``AD2.po`` gene and processing data, including ``npz`` and ``png`` . This command only needs to be entered once.

.. note:: 

   This command only needs to be used once to generate all the data of the gene fragment. If it is entered again, the previous data will be overwritten.



- ``python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t train -n AD2.po``

	- ``-m onehot_cnn_one_branch``
	
	This parameter is used to confirm the input model. A total of 11 different models can be selected, namely: ``onehot_cnn_one_branch`` / ``onehot_cnn_two_branch`` / ``onehot_embedding_dense`` / ``onehot_embedding_cnn_one_branch`` / ``onehot_embedding_cnn_two_branch`` / ``onehot_dense`` / ``onehot_resnet18`` / ``onehot_resnet34`` / ``embedding_cnn_one_branch`` / ``embedding_cnn_two_branch`` / ``embedding_dense``
	
	- ``-t train``
	
	This parameter is used to select the type of use, whether it is ``train`` or ``test`` .
	
	- ``-n AD2.po``
	
	This parameter is used to select the input gene name (the same name as the folder where the gene data is stored)

This command is used to train the onehot_cnn_one_branch model of the ``AD2.po`` gene, and the training results will be stored in ``h5_weight/AD2.po``



- ``python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n AD2.po``

	- ``-m onehot_cnn_one_branch``
	
	This parameter is used to confirm the input model. A total of 11 different models can be selected, namely: ``onehot_cnn_one_branch`` / ``onehot_cnn_two_branch`` / ``onehot_embedding_dense`` / ``onehot_embedding_cnn_one_branch`` / ``onehot_embedding_cnn_two_branch`` / ``onehot_dense`` / ``onehot_resnet18`` / ``onehot_resnet34`` / ``embedding_cnn_one_branch`` / ``embedding_cnn_two_branch`` / ``embedding_dense``
	
	- ``-t test``
	
	This parameter is used to select the type of use, whether it is ``train`` or ``test`` .
	
	- ``-n AD2.po``
	
	This parameter is used to select the input gene name (the same name as the folder where the gene data is stored)

This command is used to test the onehot_cnn_one_branch model of the ``AD2.po`` gene. It needs to store the weight data in ``h5_weight/AD2.po`` . As a result, ``auc roc`` is generated in ``log.txt`` , and the chart data is stored in the ``result`` folder.





Module Introduction
+++++++++++++++++++

This release includes multiple models,
The following are all the model names that can be entered.

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





.. image:: img/div.png