Log and Processing Tools
========================

This chapter will introduce the ``log format`` and some ``practical tools`` I wrote, these tools will reduce your data processing time and improve efficiency.

The structure of this article:

- ``log format``
- ``practical tools``
	- ``log_filter.py``
	- ``fig.py``
- ``other``
	- ``fancy_print``
	- ``gc.collect()``
	- ``getsh.py``





.. image:: img/div.png





log format
++++++++++

The log is stored in the ``log.txt`` file, and the format of the log is as follows:



.. code:: bash

	AD2.pp	onehot_cnn_one_branch	0.5202917425529678
	Sun Jun 20 12:35:02 2021
	AD2.pp	onehot_cnn_two_branch	0.5174818296823119
	Sun Jun 20 12:35:21 2021
	AD2.pp	onehot_embedding_dense	0.5079125247160411
	Sun Jun 20 12:35:37 2021
	AD2.pp	onehot_dense	0.5017822682118644
	Sun Jun 20 12:35:54 2021
	AD2.pp	onehot_resnet18	0.4990520571546792
	Sun Jun 20 12:36:13 2021
	AD2.pp	onehot_resnet34	0.49183819514395505
	Sun Jun 20 12:36:24 2021
	AD2.pp	embedding_cnn_one_branch	0.49336989913818885
	Sun Jun 20 12:36:34 2021
	AD2.pp	embedding_cnn_two_branch	0.49283976293396403
	Sun Jun 20 12:36:46 2021
	AD2.pp	embedding_dense	0.4564370769105293
	Sun Jun 20 12:36:56 2021
	AD2.pp	onehot_embedding_cnn_one_branch	0.4507754941141288
	Sun Jun 20 12:37:06 2021
	AD2.pp	onehot_embedding_cnn_two_branch	0.49117119321425656
	Sun Jun 20 12:37:40 2021
	AO.po	onehot_cnn_one_branch	0.5394717221733742
	Sun Jun 20 12:38:18 2021
	AO.po	onehot_cnn_two_branch	0.5667700186650624
	Sun Jun 20 12:38:36 2021
	AO.po	onehot_embedding_dense	0.6128413572534078
	Sun Jun 20 12:38:51 2021
	AO.po	onehot_dense	0.5894251880646438
	Sun Jun 20 12:39:07 2021
	AO.po	onehot_resnet18	0.5249458816028301
	Sun Jun 20 12:39:25 2021
	AO.po	onehot_resnet34	0.5230176726775366
	Sun Jun 20 12:39:36 2021
	AO.po	embedding_cnn_one_branch	0.5425777583760777
	Sun Jun 20 12:39:46 2021
	AO.po	embedding_cnn_two_branch	0.5162195787405871
	Sun Jun 20 12:39:58 2021
	AO.po	embedding_dense	0.4925504747353487
	Sun Jun 20 12:40:08 2021
	AO.po	onehot_embedding_cnn_one_branch	0.5016348357524828
	Sun Jun 20 12:40:18 2021
	AO.po	onehot_embedding_cnn_two_branch	0.5074866310160427

The above intercepted part of the output of ``log``, including ``AD2.pp`` gene and ``AO.po`` gene.

The following will intercept a complete log for analysis:



.. code:: bash

	Sun Jun 20 12:40:18 2021
	AO.po	onehot_embedding_cnn_two_branch	0.5074866310160427



##########

``Sun Jun 20 12:40:18 2021``

The previous content is the current ``timestamp``, which is the point in time ``when the data is written``.

To avoid ``confusion``, a ``timestamp`` has been added.

##########

``AO.po	onehot_embedding_cnn_two_branch	0.5074866310160427``

The following information is about training results.

First is the ``name of the gene``, then the ``name of the model``, and finally the ``roc auc value`` of the gene.

.. image:: img/div.png





practical tools
+++++++++++++++

I analyzed the log structure before. Log is not very easy to process. Because it contains a timestamp, I made two tools. The first tool is used to remove the timestamp and read the data into the python list, and the other tool Used to visualize graphics.



log_filter.py
-------------

.. code:: python

	f = open('log.txt', 'r')
	lines = f.readlines()
	lines = lines[1::2]

	fw = open('log_filted.txt','w+')
		
	for line in lines:
		print(line, end='')
		fw.write(line)

	fw.close()

This program will automatically delete the content of the odd-numbered lines, and write the newly arrived content into another txt file to complete the removal of the timestamp.

The data obtained is very easy to read with python.

This tool is not complicated.



fig.py
------

This tool is used to visualize the data of ``log_filted.txt``

The code of this drawing tool is shown below, and the effect is shown below

This drawing tool will automatically sort and draw the content

Because the ``log.txt`` is written in addition, there may be an out-of-order problem. This program can solve this problem.

.. code:: python

	f = open('log_filted.txt', 'r')
	lines = f.readlines()

	arr = []

	for line in lines:
		arr.append(line.split())

	name = []

	for i in arr:
		name.append(i[0])

	name = list(set(name)).sort



	for i in name:
		
		name_list = []
		num_list = []

		for j in arr:
			if j[0] == i:
				name_list.append(j[1])
				num_list.append(float(j[2]))



		import matplotlib.pyplot as plt
		import numpy as np

		plt.figure(figsize=(12, 6))
		plt.gcf().subplots_adjust(left = 0.25)

		plt.title(i)
		plt.xlabel('Area Under the Curve')
		plt.legend(labels = '')

		plt.xlim(0.4, 0.8)
		
		plt.barh(range(len(num_list)), num_list,
				 tick_label = name_list, color = plt.get_cmap('cool')(np.linspace(0, 1, 11)))
		# plt.show()
		plt.savefig('fig/' + i + '.png')

		print(i)

		# Close the currently displayed image
		plt.close()



.. image:: img/X5628FC.po.png





.. image:: img/div.png





other
+++++




fancy_print
-----------

.. code:: python

	# Nice print format
	def fancy_print(n = None, c = None, s = '#'):
		print(s * 30)
		print(n)
		print(c)
		print(s * 30)
		print() # Blank line to avoid confusion
		
This function is used to generate an eye-catching format

When the output content is too much, it is difficult to distinguish different content, so such a function is written

Some important content can be output eye-catchingly

The upper line is the name, the lower line is the content, and the user can specify the upper and lower symbols.

.. code:: python

	##############################
	model_acc
	0.856247
	##############################




gc.collect()
------------

Python has its own garbage collection mechanism, that is, the gc module

The method of use is ``import gc``

If you need python to more actively collect garbage, you can declare it at the beginning: ``gc.enable()``

Of course, you can also manually enter the command to let python collect all generation garbage: ``gc.collect()``




getsh.py
--------

This small tool is used to ``automatically`` generate ``sh`` scripts, so that there is no need to manually edit a lot of data.





.. image:: img/div.png