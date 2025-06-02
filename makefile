test_example:
	python3 -m pytest --print tests/test_inception.py

kaggle_example:
	PYTHONPATH=. python3 kaggle/MNIST/gan_train.py


