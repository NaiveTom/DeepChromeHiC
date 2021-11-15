.. image:: img/data-center-carbonate-HERO.jpg





|





Run on a Supercomputer
======================

DeepChromeHiC consumes computing resources very much and generates a lot of data. The program can be run on a supercomputer.

This chapter will introduce how to run DeepChromeHiC on a supercomputer.





Supercomputer using the Ubantu system
+++++++++++++++++++++++++++++++++++++



1. Copy ``DeepChromeHiC`` to your supercomputer directory



2. Execute the command to ``DeepChromeHiC`` directory

	.. code:: bash

		cd D:\DeepChromeHiC
	
	
	
3. Create a new folder called ``your gene name`` in the ``data`` folder, for example, ``AD2.po``.

	Then paste your genetic data into this folder. 

	- ``seq.anchor1.pos.txt`` 

		Put in the enhancer positive fragments

	- ``seq.anchor1.neg2.txt`` 

		Put in the enhancer negative fragments

	- ``seq.anchor2.pos.txt`` 

		Put in the promoter positive fragments

	- ``seq.anchor2.neg2.txt`` 

		Put in the promoter negative fragments



4. Use python3 to preprocess the data so that train and test can be run later

	.. code:: bash

		python3 DeepChromeHiC.py -p true -n AD2.po

	Then it will automatically start pre-processing AD2.po.

	This step will generate png files and npz files.



5. Use python3 to run DeepChromeHiC-train, here is AD2.po as an example

	.. code:: bash

		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_dense -t train -n AD2.po

		python3 DeepChromeHiC.py -m onehot_dense -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet18 -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet34 -t train -n AD2.po

		python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m embedding_dense -t train -n AD2.po

		python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t train -n AD2.po
	
	
	
6. Use python3 to run DeepChromeHiC-test, here is AD2.po as an example

	.. code:: bash

		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n AD2.po

		python3 DeepChromeHiC.py -m onehot_dense -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet34 -t test -n AD2.po

		python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m embedding_dense -t test -n AD2.po

		python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n AD2.po
	
	
	
7. Get results

	All picture results are saved in the ``result`` folder, for convenience, the text results are all in ``log.txt``

	``log.txt`` adopts the append mode, which means that the previous results will always remain in the ``log.txt`` file and will not be deleted due to the running program. If you want to re-record the output results, then you need to delete the ``log.txt`` folder.





.. image:: img/div.png





Supercomputer using the Slurm job system
++++++++++++++++++++++++++++++++++++++++

If the supercomputer you use is publicly used on campus, it is likely to use the slurm job system or the PBS job system to manage all jobs.

To use the slurm job system or the PBS job system, you need to write .sh scripts to arrange jobs.



1. Copy ``DeepChromeHiC`` to your supercomputer directory



2. Execute the command to ``DeepChromeHiC`` directory

	.. code:: bash

		cd D:\DeepChromeHiC
	
	
	
3. Create a new folder called ``your gene name`` in the ``data`` folder, for example, ``AD2.po``.

	Then paste your genetic data into this folder. 

	- ``seq.anchor1.pos.txt`` 
	
		Put in the enhancer positive fragments

	- ``seq.anchor1.neg2.txt`` 
	
		Put in the enhancer negative fragments

	- ``seq.anchor2.pos.txt`` 
	
		Put in the promoter positive fragments

	- ``seq.anchor2.neg2.txt`` 
	
		Put in the promoter negative fragments



4. Use python3 to preprocess the data so that train and test can be run later:

	Write a slurm script into the ``slurm`` folder, for example ``prep_AD2.po.sh``

	You can use the ``vim`` tool to write directly in Linux or use tools such as ``winscp`` to copy scripts written on Windows to your linux, but please note that in Windows, use ``\r\n`` for line breaks, but for linux use ``\n`` So you need to use a tool such as ``Notepad++`` to delete (replace it with empty) ``\r``, otherwise linux will report an error.

	.. code:: bash

		#!/bin/bash

		#SBATCH -J prep_AD2.po
		#SBATCH -p general
		#SBATCH -o log/prep_AD2.po_%j.log
		#SBATCH -e log/prep_AD2.po_%j.err
		#SBATCH --mail-type=ALL
		#SBATCH --mail-user=Write your work email here!
		#SBATCH --nodes=1
		#SBATCH --ntasks-per-node=1
		#SBATCH --cpus-per-task=1
		#SBATCH --time=24:00:00
		#SBATCH --mem=64G

		module load deeplearning/2.3.0

		cd ..

		python3 DeepChromeHiC.py -p true -n AD2.po
		
		
	
	
	
.. topic:: Description

	- ``#!/bin/bash``
	
		This is a bash script, so linux can recognize the text content

	- ``#SBATCH -J prep_AD2.po``
	
		Let's give this script a name so that we can see the name when we look at the task later. Please modify this name so that it can be clearly displayed in the task list

	- ``#SBATCH -p general``
	
		On the supercomputer I use, tasks that use cpu use the general node, so general is written here. If you need more information, please consult the slurm documentation or check your school's supercomputer usage plan.

	- ``#SBATCH -o log/prep_AD2.po_%j.log``
	
		When sulrm is running, nothing will be output to the screen, so a document is needed to save the logs and errors during slurm running, so that it can be checked later. Here is the log file. The log will be output to ``log/prep_AD2.po_%j.log``, which is the ``prep_AD2.po_%j.log`` file under the ``log`` folder, where ``%j`` is the task number assigned to you by slurm.

	- ``#SBATCH -e log/prep_AD2.po_%j.err``
	
		When sulrm is running, nothing will be output to the screen, so a document is needed to save the logs and errors during slurm running, so that it can be checked later. Here is the error file. Errors and prompts will be output to ``log/prep_AD2.po_%j.err``, which is the ``prep_AD2.po_%j.err`` file under the ``log`` folder, where ``%j`` is the task number assigned to you by slurm.

	- ``#SBATCH --mail-type=ALL``
	
		All emails will be sent to you, including task start email, task end email, and task error email. These emails contain various information, including how long your task has been performed, how long you have waited in the team, cpu usage, memory usage, gpu usage, etc.

	- ``#SBATCH --mail-user=Write your work email here!``
	
		Replace ``Write your work email here!`` with your email address so that you can receive emails.

	- ``#SBATCH --nodes=1``
	
		The number of nodes used by the task. Because python runs on a single core, you can write 1 directly here

	- ``#SBATCH --ntasks-per-node=1``
	
		The number of nodes used by the task. Because python runs on a single core, you can write 1 directly here

	- ``#SBATCH --cpus-per-task=1``
	
		The number of nodes used by the task. Because python runs on a single core, you can write 1 directly here

	- ``#SBATCH --time=24:00:00``
	
		The maximum running time, write as long as possible, if the running time is exceeded and the running is not completed, then the task will be ended directly, which is what we don’t want to see

	- ``#SBATCH --mem=64G``
	
		Run the memory size, write as large as possible, if possible, please keep the content size of 64GB, if the memory usage exceeds the maximum memory, the task will fail directly. This is what we don't want to see.

	- ``module load deeplearning/2.3.0``
	
		If the system you are using is ``redhat`` and module is used to manage system ``modules``, then please load modules containing python3 and various libraries so that it can run normally in the future.

	- ``cd ..``
	
		Because all slurm scripts are in the slurm folder, this is the root directory of this program. In order to access ``DeepChromeHiC.py`` normally, you need to jump to the upper level directory. The purpose here is even so

	- ``python3 DeepChromeHiC.py -p true -n AD2.po``
	
		Use python3 to run DeepChromeHiC.py to prepare the ``png`` file and ``npz`` file of ``AD2.po``.





	.. note::

		Replace ``Write your work email here!`` with your work email
	
	
	
5. Run the slurm script

	.. code:: bash

		cd DeepChromeHiC
		cd slurm
		sbatch prep_AD2.po.sh

	Enter your ``slurm`` folder and run the script ``prep_AD2.po.sh`` written before
	
	
	
6. Use python3 to run DeepChromeHiC-train, here is AD2.po as an example

	We need to write the slurm script first and then run the script. In this step, we first write the slurm script.

	.. code:: bash

		#!/bin/bash

		#SBATCH -J train_AD2.po
		#SBATCH -p dl
		#SBATCH --gres=gpu:v100:1
		#SBATCH -o log/train_AD2.po_%j.log
		#SBATCH -e log/train_AD2.po_%j.err
		#SBATCH --mail-type=ALL
		#SBATCH --mail-user=Write your work email here!
		#SBATCH --nodes=1
		#SBATCH --ntasks-per-node=1
		#SBATCH --time=48:00:00
		#SBATCH --mem=32G

		module load deeplearning/2.3.0

		cd ..

		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_dense -t train -n AD2.po

		python3 DeepChromeHiC.py -m onehot_dense -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet18 -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet34 -t train -n AD2.po

		python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m embedding_dense -t train -n AD2.po

		python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t train -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t train -n AD2.po
		
		
		
	
	
.. topic:: Description

	Here I only write the parts that are different from the previous step. For the same parts, please refer to the previous part.

	- ``#SBATCH -p dl``
	
		Here we use a node with GPU. On my supercomputer, it is ``dl`` (abbreviation for deeplearning). Please refer to the name of your supercomputer node and replace it with your supercomputer node.

	- ``#SBATCH --gres=gpu:v100:1``
	
		The GPU is used here, and the GPU name must be specified in slurm. Here, a NVIDIA v100 GPU is used. Please note that multiple GPUs should not be used here unless the GPU is virtualized. In addition, do ``not`` use a GPU with a video memory size of ``less than 8GB``. Which can not finish the entire program.

	- ``#SBATCH --time=48:00:00``
	
		Please increase the running time as much as possible, because this procedure is more time-consuming than the previous prep procedure. If the time is too short, it may not be able to complete the whole procedure with the operation.

	- ``#SBATCH --mem=32G``
	
		Please specify the memory size, generally ``8GB`` is sufficient, but if the gene fragment is too large, please use more memory (python3 memory leak problem is bad)

	- ``python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t train -n AD2.po``
	
		Here we directly run 11 different models. Generally speaking, it takes between 3-10 hours for a gene fragment of normal size.





	.. note::

		Replace ``Write your work email here!`` with your work email



7. Run the slurm script to execute the training program

	.. code:: bash

		cd DeepChromeHiC
		cd slurm
		sbatch train_AD2.po.sh

	Enter your ``slurm`` folder and run the script ``train_AD2.po.sh`` written before



8. Use python3 to run DeepChromeHiC-test, here is AD2.po as an example

	We need to write a slurm script to arrange the job and run the test program.

	.. code:: bash

		#!/bin/bash

		#SBATCH -J test
		#SBATCH -p dl
		#SBATCH --gres=gpu:v100:1
		#SBATCH -o log/test_%j.log
		#SBATCH -e log/test_%j.err
		#SBATCH --mail-type=ALL
		#SBATCH --mail-user=Write your work email here!
		#SBATCH --nodes=1
		#SBATCH --ntasks-per-node=1
		#SBATCH --time=48:00:00
		#SBATCH --mem=64G

		module load deeplearning/2.3.0

		cd ..





		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n AD2.po

		python3 DeepChromeHiC.py -m onehot_dense -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_resnet34 -t test -n AD2.po

		python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m embedding_dense -t test -n AD2.po

		python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n AD2.po
		python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n AD2.po
		
		
		
		
		
		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n AO.po
		python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n AO.po
		python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n AO.po

		python3 DeepChromeHiC.py -m onehot_dense -t test -n AO.po
		python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n AO.po
		python3 DeepChromeHiC.py -m onehot_resnet34 -t test -n AO.po

		python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n AO.po
		python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n AO.po
		python3 DeepChromeHiC.py -m embedding_dense -t test -n AO.po

		python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n AO.po
		python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n AO.po





		python3 DeepChromeHiC.py -m onehot_cnn_one_branch -t test -n BL1.po
		python3 DeepChromeHiC.py -m onehot_cnn_two_branch -t test -n BL1.po
		python3 DeepChromeHiC.py -m onehot_embedding_dense -t test -n BL1.po

		python3 DeepChromeHiC.py -m onehot_dense -t test -n BL1.po
		python3 DeepChromeHiC.py -m onehot_resnet18 -t test -n BL1.po
		python3 DeepChromeHiC.py -m onehot_resnet34 -t test -n BL1.po

		python3 DeepChromeHiC.py -m embedding_cnn_one_branch -t test -n BL1.po
		python3 DeepChromeHiC.py -m embedding_cnn_two_branch -t test -n BL1.po
		python3 DeepChromeHiC.py -m embedding_dense -t test -n BL1.po

		python3 DeepChromeHiC.py -m onehot_embedding_cnn_one_branch -t test -n BL1.po
		python3 DeepChromeHiC.py -m onehot_embedding_cnn_two_branch -t test -n BL1.po
		
		
		
		
		
		.
		.
		.
		.
		.
		
	The time required for the test is relatively short, about 30 minutes for each gene fragment, so you can test multiple fragments at a time, just as written in my program, I tested three gene fragments at a time. Note that python3 has a memory leak problem, so more memory is needed to ensure that the program can run normally.
		
		
		
		
		
	.. note::

		Replace ``Write your work email here!`` with your work email
		
	
	
9. Get results

	All picture results are saved in the ``result`` folder, for convenience, the text results are all in ``log.txt``

	``log.txt`` adopts the append mode, which means that the previous results will always remain in the ``log.txt`` file and will not be deleted due to the running program. If you want to re-record the output results, then you need to delete the ``log.txt`` folder.

	Using slurm can quickly detect multiple genes. For example, this supercomputer can run 30 CPU scripts and 4 GPU scripts at a time. Using slurm can save the idle time between programs.





Supercomputer using the PBS job system
++++++++++++++++++++++++++++++++++++++

The use of PBS script and Slurm script are very similar, so I won't go into details here. For resource application, please refer to Slurm resource application.





.. image:: img/div.png