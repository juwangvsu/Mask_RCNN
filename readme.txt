jupyter in .local/bin/ is no good, remove .local/bin from PATH, jupyter
installed in system will work.

cd samples$ jupyter notebook
click demo.ipynb, run through a few steps and it will download mask_rcnn_coco.h5

---------evaluate on dataset ------------------------------------------------
python3 samples/coco/coco.py evaluate --dataset=dog2.jpg --model=mask_rcnn_coco.h5
	gpu out of memory if jupyter still running.
	dataset error

----------------1/11/2020 training test with balloon-------------------------------------------------------------
python3 balloon.py train --dataset=/media/student/code1/rcnn/Mask_RCNN/balloon --weights=coco
	Loading weights  /media/student/code1/rcnn/Mask_RCNN/mask_rcnn_coco.h5, 
	start with the pretrained coco weight, loss 0.6 at epoch 1, 0.14 at epoch 29
	result is pretty good just after one epoch.

----------------1/10/2020 training-------------------------------------------------------------
Mask_RCNN/samples/coco$ 
s1:	python3 coco.py train --dataset=/media/student/coco --model=coco --download=True
		train from pretrained coco model mask_rcnn_coco.h5
s2:	python3 coco.py train --dataset=/media/student/coco --model=none
		train from scratch
s3:	python3 coco.py train --dataset=/media/student/coco --model=/media/student/code1/rcnn/Mask_RCNN/logs/coco20200110T1606/mask_rcnn_coco_0068.h5
		continue train from an .h5 file

this download coco data set (big) and train.

bugs:
	'Model' object has no attribute 'metrics_tensors'
	fix: changing: self.keras_model.metrics_tensors.append(loss) to self.keras_model.add_metric(loss, name)

	run out of gpu memory.
	fix: coco.py,
		IMAGES_PER_GPU = 1

train performance:
   s3:msi
	stage 1: loss: 2.0799 - rpn_class_loss: 0.1643
		11 hrs, not much improve
train results:
	logs/
----------------1/9/2020 simple detect-------------------------------------------------------------
detect test:
	copy code from demo.ipynb to cocotest.py and make some modification
	cd samples
	sudo python3 -m pip install -U Pillow
	python3 cocotest.py
	python3 cocotest.py --model logs/coco20200110T1305/mask_rcnn_coco_0002.h5
	result: good

        detect time:  0.4716019630432129 seconds
        keras_model.predict(): 0.34 seconds
Total params: 64,158,584
Trainable params: 64,047,096
Non-trainable params: 111,488
number of layers: 394


