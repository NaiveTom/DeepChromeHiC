test.py
=======

In order to adapt to different models, ``test.py`` is divided into three parts to perform three different model training tasks. test.py is friendly to supercomputer logs. It turns off the display of the progress bar to avoid flooding the screen and log system. At the same time, the picture will not be displayed and will be saved directly to the corresponding location.





.. image:: img/div.png





test_cnn_dense_resnet
+++++++++++++++++++++



Function prototype
------------------

.. code-block:: python

	void test_cnn_dense_resnet(gen_name, model_name)



This function is called by ``DeepChromeHiC.py`` . There are two inputs: the name of the gene to be trained ``gen_name`` and the name of the selected model ``model_name``

This function has no return value.

This function can test the following models:

#. ``onehot_cnn_one_branch``
#. ``onehot_embedding_dense``
#. ``onehot_dense``
#. ``onehot_resnet18``
#. ``onehot_resnet34``

The model used by this function does ``not`` require ``dna2vec embedding``, and the data is ``merged before entering the model``.



ImageDataGenerator
------------------

In order to reduce the problem of storage space consumption, this model uses ``png`` images as input, and png uses Huffman coding compression, which greatly reduces the volume of data. Because the CPU speed is faster than the disk, it will not consume too much time to decode after being compressed into png format, and it is faster than reading the original data under the actual test.

ImageDataGenerator should not shuffle the data.



.. code-block:: python

	from keras.preprocessing.image import ImageDataGenerator

	datagen = ImageDataGenerator(rescale = 1./255)

    BATCH_SIZE = 32

    test_generator = datagen.flow_from_directory(directory = 'data/'+gen_name+'/png_test/', target_size = (20002, 5),
                                                 color_mode = 'grayscale',
                                                 batch_size = BATCH_SIZE,
                                                 shuffle = False) # do not shuffle



Load Weights
-------------

This function needs to ``load`` the previously trained ``weights`` and read the model in ``model.py``, so that the program can run.



.. code-block:: python

	clf.load_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')

    fancy_print('Load the model ...', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')



Show Auc Roc Curve
------------------

This project uses the ``auc roc curve`` for ``evaluation``, and you can add other evaluation functions to ``test.py``, such as ``accuracy``, ``f1``, ``entropy``, ``loss``, and so on.

This project can automatically calculate ``auc`` and use ``matplotlib`` to draw ``roc curve``.



.. code-block:: python
	
	# Calculate ROC curve and AUC for each category
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Two classification problem
    n_classes = label_test.shape[1] # n_classes = 2
    fancy_print('n_classes', n_classes) # n_classes = 2

    # Draw ROC curve using actual category and predicted probability
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fancy_print('roc_auc', roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color = 'darkorange', 
             lw = lw, label = 'ROC curve (area = %0.6f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc = "lower right")

    plt.savefig('result/'+gen_name+'/'+model_name+'/rocauc.png')
    # plt.pause(0.01)
    # plt.show() # Supercomputers are completely command lines and cannot display pictures.



Save Result as Text
-------------------

In order to facilitate reading and subsequent processing, and to know the ``output time`` of each training (which can be determined according to the output time), the content is written into a ``txt`` file.

The file adopts the scheme of append writing, so that the previous data will not be lost.



.. code-block:: python

	fw = open('log.txt','a+')
    import time
    fw.write( time.asctime(time.localtime(time.time())) + '\n' )
    fw.write( gen_name + '\t' + model_name + '\t' + str(roc_auc[1]) + '\n' )
    fw.close()





.. image:: img/div.png





test_cnn_separate
+++++++++++++++++



Function prototype
------------------

.. code-block:: python

	void test_cnn_separate(gen_name, model_name)



This function is called by ``DeepChromeHiC.py`` . There are two inputs: the name of the gene to be trained ``gen_name`` and the name of the selected model ``model_name``

This function has no return value.

This function can test the following models:

#. ``onehot_cnn_two_branch``

The model used by this function does ``not`` require ``dna2vec embedding``, and the data is ``not merged before entering the model``.



ImageDataGenerator
------------------

In order to reduce the problem of storage space consumption, this model uses ``png`` images as input, and png uses Huffman coding compression, which greatly reduces the volume of data. Because the CPU speed is faster than the disk, it will not consume too much time to decode after being compressed into png format, and it is faster than reading the original data under the actual test.

ImageDataGenerator should not shuffle the data.



.. code-block:: python

	from keras.preprocessing.image import ImageDataGenerator

    BATCH_SIZE = 32

    test_datagen = ImageDataGenerator(rescale = 1. / 255)

    # This is used to generate the label
    test_generator = test_datagen.flow_from_directory(directory = 'data/'+gen_name+'/test_en/', target_size=(10001, 5),
                                                      color_mode = 'grayscale',
                                                      class_mode = 'categorical',
                                                      batch_size = BATCH_SIZE,
                                                      shuffle = False) # do not shuffle

    def generator_two_test():
        test_generator1 = test_datagen.flow_from_directory(directory = 'data/'+gen_name+'/test_en/', target_size = (10001, 5),
                                                          color_mode = 'grayscale',
                                                          class_mode = 'categorical', 
                                                          batch_size = BATCH_SIZE,
                                                          shuffle = False) # do not shuffle

        test_generator2 = test_datagen.flow_from_directory(directory = 'data/'+gen_name+'/test_pr/', target_size = (10001, 5),
                                                          color_mode = 'grayscale',
                                                          class_mode = 'categorical',
                                                          batch_size = BATCH_SIZE,
                                                          shuffle = False) # do not shuffle
        while True:
            out1 = test_generator1.next()
            out2 = test_generator2.next()
            yield [out1[0], out2[0]] # , out1[1] # Return the combination of two and the result



Load Weights
-------------

This function needs to ``load`` the previously trained ``weights`` and read the model in ``model.py``, so that the program can run.



.. code-block:: python

	clf.load_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')

    fancy_print('Load the model ...', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')



Show Auc Roc Curve
------------------

This project uses the ``auc roc curve`` for ``evaluation``, and you can add other evaluation functions to ``test.py``, such as ``accuracy``, ``f1``, ``entropy``, ``loss``, and so on.

This project can automatically calculate ``auc`` and use ``matplotlib`` to draw ``roc curve``.



.. code-block:: python
	
	# Calculate ROC curve and AUC for each category
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Two classification problem
    n_classes = label_test.shape[1] # n_classes = 2
    fancy_print('n_classes', n_classes) # n_classes = 2

    # Draw ROC curve using actual category and predicted probability
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fancy_print('roc_auc', roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color = 'darkorange', 
             lw = lw, label = 'ROC curve (area = %0.6f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc = "lower right")

    plt.savefig('result/'+gen_name+'/'+model_name+'/rocauc.png')
    # plt.pause(0.01)
    # plt.show() # Supercomputers are completely command lines and cannot display pictures.



Save Result as Text
-------------------

In order to facilitate reading and subsequent processing, and to know the ``output time`` of each training (which can be determined according to the output time), the content is written into a ``txt`` file.

The file adopts the scheme of append writing, so that the previous data will not be lost.



.. code-block:: python

	fw = open('log.txt','a+')
    import time
    fw.write( time.asctime(time.localtime(time.time())) + '\n' )
    fw.write( gen_name + '\t' + model_name + '\t' + str(roc_auc[1]) + '\n' )
    fw.close()





.. image:: img/div.png





test_embedding
++++++++++++++



Function prototype
------------------

.. code-block:: python

	void test_embedding(gen_name, model_name)



This function is called by ``DeepChromeHiC.py`` . There are two inputs: the name of the gene to be trained ``gen_name`` and the name of the selected model ``model_name``

This function has no return value.

This function can test the following models:

#. ``onehot_cnn_two_branch``

The model used by this function require ``dna2vec embedding``.



Load Weights
-------------

This function needs to ``load`` the previously trained ``weights`` and read the model in ``model.py``, so that the program can run.



.. code-block:: python

	clf.load_weights('h5_weights/'+gen_name+'/'+model_name+'.h5')

    fancy_print('Load the model ...', 'h5_weights/'+gen_name+'/'+model_name+'.h5', '=')



Show Auc Roc Curve
------------------

This project uses the ``auc roc curve`` for ``evaluation``, and you can add other evaluation functions to ``test.py``, such as ``accuracy``, ``f1``, ``entropy``, ``loss``, and so on.

This project can automatically calculate ``auc`` and use ``matplotlib`` to draw ``roc curve``.



.. code-block:: python
	
	# Calculate ROC curve and AUC for each category
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Two classification problem
    n_classes = label_test.shape[1] # n_classes = 2
    fancy_print('n_classes', n_classes) # n_classes = 2

    # Draw ROC curve using actual category and predicted probability
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fancy_print('roc_auc', roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color = 'darkorange', 
             lw = lw, label = 'ROC curve (area = %0.6f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc = "lower right")

    plt.savefig('result/'+gen_name+'/'+model_name+'/rocauc.png')
    # plt.pause(0.01)
    # plt.show() # Supercomputers are completely command lines and cannot display pictures.



Save Result as Text
-------------------

In order to facilitate reading and subsequent processing, and to know the ``output time`` of each training (which can be determined according to the output time), the content is written into a ``txt`` file.

The file adopts the scheme of append writing, so that the previous data will not be lost.



.. code-block:: python

	fw = open('log.txt','a+')
    import time
    fw.write( time.asctime(time.localtime(time.time())) + '\n' )
    fw.write( gen_name + '\t' + model_name + '\t' + str(roc_auc[1]) + '\n' )
    fw.close()





.. image:: img/div.png