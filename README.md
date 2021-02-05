# MNIST---Handwritten-Digit-Classification-by-CNN

The MNIST database of handwritten digits consists of 60000 training images and 10000 testing images. It is a well-known database in image classification and similar fields. An artificial neural network is designed and trained to classifies the MNIST database. The developed neural network is based on convolutional neural networks (CNN). The CNNs are selected because the task is image classification, and the designed network would deal with images. The convolutional based neural networks can capture and gather more information from images in comparison with other types of neural networks. The MNIST database consists of 70000 images. The dimensional of the images are 28 x 28 pixels. The images are labeled by a digit from 0 to 9 according to their content. The problem is training and testing an artificial neural network by the MNIST database. As aforementioned, the designed neural network is based on the convolutional blocks. In addition, the pooling layers, dense layers, and fully connected layers are also utilized. The designed network consists of 6 layers. The first four layers are convolutional-based. The max-pooling layers follow the convolutional layers two and four. The fifth layer is a flatten layer that makes the convolutional layer (fourth layer) to a flatten layer. The sixth layer is a fully connected layer whose dropout is set to 0.2 to avoid overfitting. The last layer is the output layer which its dimension is 10, due to the number of the classes (the digits from 0 to 9). Additionally, for each of the layers, batch normalization has been utilized to improve the performance of the training process. A summary of the network architecture is provided in table 1. For the training process, the “ImageDataGenerator” is also utilized to generate the augmented image data.

                                                  Table 1: The summary of network architecture

   ![Untitled55](https://user-images.githubusercontent.com/30000556/107068237-99357d00-67f5-11eb-9c40-a21785f98b53.png)

The loss function of the designed network is set to the categorical cross-entropy, and the Adam optimizer is used for the optimization. As mentioned, the MNIST database has 60000 training images (data sample). It has approximately equal distribution among the different classes (uniform distribution). According to the computational loading, the batch size and number of epochs are selected to 1024 and 20, respectively. The accuracy and loss values for the training data and the validation data in the training step for each epoch are shown in figure 1.

![dsdc](https://user-images.githubusercontent.com/30000556/107068514-fa5d5080-67f5-11eb-9fcc-370cdf5ad280.png)

![dsdc4444](https://user-images.githubusercontent.com/30000556/107068579-0d702080-67f6-11eb-9fe6-603fa6581586.png)


                                                         Figure 1: the model accuracy and model loss curves

As shown in figure 1, the accuracy and loss values have similar behavior, meaning that the designed neural network avoids overfitting to the training data.  
The results are provided in the following. The total accuracy for the testing data is 99.5%. The accuracy for each class is presented in the following:

                                                            Table 2: The accuracy for each Class
Class	Accuracy	Class	Accuracy
0	100.00	5	99.21
1	99.64	6	99.26
2	99.80	7	99.22
3	99.81	8	99.28
4	99.38	9	99.30

The precision, recall and F1 score for the each of the classes are presented in table 3. The confusion matrix is provided in the table4.

![Untitled88](https://user-images.githubusercontent.com/30000556/107068957-81122d80-67f6-11eb-80ce-bc457c3a2576.png)

               Table 3: the precision, recall, and F1-score of each class (the fourth column indicates the number of samples is each class)

![Untitled77](https://user-images.githubusercontent.com/30000556/107068931-7788c580-67f6-11eb-9928-7b1df199e4c2.png)

                                                               Table 4: The Confusion Matrix 
                      
                               

As the results show, the designed network has adequate performance for each of the classes and also has not been overfitted to the training data (according to figure 1 curves). As you can see in the last column of Table 2, the distribution is approximately equal among the different classes. (The distribution of the training data is also close to the distribution of the testing data). In table 3, each row indicates the actual class, and its corresponding column indicates the predicted class. The multi-layer network is designed to gain as much information as possible from the database. In a multi-layer network (deep network), the deeper layers can recognize the more complicated pattern in comparison to the first layers. Therefore, by utilizing the multi-layer network, the performance would be improved as the results illustrate in this case.
