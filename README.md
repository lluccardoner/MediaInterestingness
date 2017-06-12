# Deep Learning For Multimedia Processing - Predicting Media Interestingness
MediaEval task of 2017 as bachelors thesis.
http://www.multimediaeval.org/mediaeval2017/mediainterestingness/
# Author
Lluc Cardoner Campi

lluccardoner@gmail.com

## Abstract
This thesis explores the application of a deep learning aproach for the prediction of media interestingness. Two different models are investigated, one for the prediction of image and one for the prediction of video interestingness.

For the prediction of image interestingness, the \textit{ResNet50} network is fine-tuned to obtain the best results. First, some layers are added. Next, the model is trained and fine-tuned using data augmentation, dropout, class weights, and changing other hyper parameters.

For the prediction of video interestingness, first, features are extracted with a 3D convolutional network. Next a LSTM network is trained and fine-tuned with the features. 

The final result is a binary label for each image/video: 1 for interesting, 0 for not interesting. Additionally, a confidence value is provided for each prediction. Finally, the Mean Average Precision (MAP) is employed as evaluation metric to estimate the quality of the final results.