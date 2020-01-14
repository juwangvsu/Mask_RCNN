jupyter in .local/bin/ is no good, remove .local/bin from PATH, jupyter
installed in system will work.

cd samples$ jupyter notebook
click demo.ipynb, run through a few steps and it will download mask_rcnn_coco.h5

---------evaluate on dataset ------------------------------------------------
/media/student/code1/rcnn/Mask_RCNN$ 
	python3 samples/coco/coco.py evaluate --dataset=/media/student/coco --model=/media/student/coco/logs/coco20200112T1801/mask_rcnn_coco_0112.h5 --limit=10

gpu out of memory if jupyter still running.

mask_rcnn_coco_0001.h5 (locally trained, epoch 1)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.003
mask_rcnn_coco_0160.h5
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.435
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.139
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.032
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.157
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
mask_rcnn_coco.h5 (pretrained weights)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.694
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.509
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.139
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.738

----------------1/12/2020 training test with coco -------------------------------------------------------------
	from --model=imagenet , space out at 65 epoch, continue from 65 checkpoint
   1/12/2020, homepc, checkpoint interval 8
	Epoch 94/120 loss: 1.3929

----------------1/11/2020 training test with balloon-------------------------------------------------------------
python3 balloon.py train --dataset=/media/student/code1/rcnn/Mask_RCNN/balloon --weights=coco
	Loading weights  /media/student/code1/rcnn/Mask_RCNN/mask_rcnn_coco.h5, 
	start with the pretrained coco weight, loss 0.6 at epoch 1, 0.14 at epoch 29
	result is pretty good just after one epoch.

----------------1/10/2020 training-------------------------------------------------------------
msi
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
   1/12/2020, homepc, checkpoint interval 8,mrcnn/model.py
	keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True,period=8),

	Epoch 94/120 loss: 1.3929
   homepc, stock weights
	epoch 1, loss 0.86, start at 0.3
   homepc, imagenet
	epoch 160, loss 1.2790 
 
train results:
	logs/  this point to one of the logs-xxxx
	logs-mrcnn-imagenet train result with model=imagenet
	logs-mrcnn-stock train result with model=mask_rcnn_coco.h5, the pretrained weights

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


