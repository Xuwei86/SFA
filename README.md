SFA: An Attention Mechanism for Remarkable CNN Performance Improvement in Diverse Vision Tasks.

Ths project of SFA attention module, include object detection(MS COCO and Pascal VOC), ImageNet classification(MobileNetv2 as baseline) and semantic segmentation (deeplabv3 as the base line)

This is the structure of the SFA
[[![Uploading arch1.png…]()
](https://github.com/Xuwei86/SFA/blob/main/arch1.png)](https://github.com/Xuwei86/SFA/blob/main/arch1.png)

1、For ImageNet classification, the first choice needs to import the ImageNet dataset (we do not provide the dataset here, note the path).
Training
(Dual GPU or above is recommended)
Execute the command: CUDA_VISIBLE_DEVICES=0,1 python imagenet.py -a mbv2_sea_attention -b 64 -d ....

2, for Object detection (all need to import the dataset, pay attention to add the path)
1) Pay attention to select the model to use, in the train.py file, for example, field phi = 'nano' for the selection of YOLOX-Nano model
2) To use MS COCO and Pascal VOC, you need to modify train_annotation_path, val_annotation_path and classes_path according to their respective datasets. their labeling files are provided.
3) After setting the hyperparameters in the train.py file,
    Execute the command: python train.py.


3、Semantic Segmentation (all need to import the dataset, pay attention to add the path)
After setting the hyperparameters,  
Execute the command:  python main.py; of course, the hyperparameters can also be added in the command.
