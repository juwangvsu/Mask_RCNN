jupyter in .local/bin/ is no good, remove .local/bin from PATH, jupyter
installed in system will work.

cd samples$ jupyter notebook
click demo.ipynb, run through a few steps and it will download mask_rcnn_coco.h5

python3 samples/coco/coco.py evaluate --dataset=dog2.jpg --model=mask_rcnn_coco.h5
	gpu out of memory if jupyter still running.
	dataset error

detect test:
	copy code from demo.ipynb to cocotest.py and make some modification
	cd samples
	sudo python3 -m pip install -U Pillow
	python3 cocotest.py
	result: good.
