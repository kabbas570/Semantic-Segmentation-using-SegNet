# Semantic-segmentatio using-SegNet


SegNet is a deep encoder-decoder architecture for multi-class pixelwise segmentation researched and developed by members of the Computer Vision and Robotics Group at the University of Cambridge, UK. The demo above is an example of using CNN for your own custom dataset.

![image](https://user-images.githubusercontent.com/56618776/91529826-f70b9200-e944-11ea-8aa9-4b1aa72c98dc.png)

 SegNet uses the max pooling indices to upsample (without learning) the feature map(s) 
 
 
![image](https://user-images.githubusercontent.com/56618776/91559698-8465db00-e973-11ea-82b7-2e0c84da4d82.png)

 The input for the network is an RGB Image and target is a binary mask of each class concatinated in a tensor.

## Input Imgae                                         

![image](https://user-images.githubusercontent.com/56618776/91530231-bc562980-e945-11ea-90b1-d1d7cb2f64dd.png)        
## Target

 ![image](https://user-images.githubusercontent.com/56618776/91530411-ff180180-e945-11ea-9d1a-b272b0931a38.png)


### The orignal paper of the SegNet architecture is :

Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495.


