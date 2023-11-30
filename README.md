# ai_application
see [documentation.ipynb](./documentation.ipynb)

When cloning the repository use the --recursive flag to include the submodule contents

```shell
$ git clone https://github.com/ftrtz/ai_application.git --recursive
```

## Run main.py with docker to generate and store results

Build docker image

```shell
$ docker build --tag ai_application .
```

Start docker container

```shell
$ docker-compose up
```

Results are stored in [/results](./results)

## AI Coding Challenge
**Task 1:** The first task concentrates on the implementing CNN architecture and model training; how to the manage repositories in version control.

1. Make two repositories named as follows: ‘ai_library’, ‘ai_application’. Perform the following tasks:
	1. In ai_library, create script/s implementing your own custom CNN architecture (for eg. 5 Conv layers, 2 Dense layers, use Batch normalization layers and non-linear layers wherever appropriate) using TensorFlow/Pytorch. The script should be implemented in object-oriented fashion (using classes, public-private methods) and follow PEP8 coding protocols. Use Adam optimizer and any loss function of your choice.
	2. Create a training pipeline using custom generated data i.e. images of size 64x64 that contain different shapes such as circle, square, rectangle etc. and groundtruth as the shape classification; integrate with the custom CNN implemented in 1a. Create enough data so that we have train-test-evaluation split
	3. Bonus: Import Tensorboard to visualize loss, accuracy curves.
	4. Train a model and predict on the evaluation data. How would you evaluate the performance of your model? Select appropriate evaluation metrices and provide the result.

**Task 2:** The second task deals with utilizing the inheritance concept of object-oriented programming.

2. In the repository ‘ai_application’, add the repo ‘ai_library’ as the submodule. Perform the following tasks:
	1. Create a script that inherits the data generation class from ai_library submodule and overwrite custom data generation replacing with the publicly available MNIST handwritten digits dataset instead.
	2. Train the model with this new dataset (you would have to overwrite the CNN architecture output size accordingly).
	3. Bonus: Add the trained model to the ai_application repository as Git LFS file.

**Task 3:** A peek into dockerization in the third step which allows any project to be easily deployable and portable across platforms.

3. Create a simple Dockerfile in the repo ‘ai-application’ that performs the following tasks:
	1. Build a docker image that installs python and all dependencies required for training.
	2. Copy the project ‘ai_application’ to the docker image with the trained model and evaluation data.
	3. Run the container so that it executes the prediction script and generates the results.
	4. Bonus: Apply volume mapping, so that result saved in the container is copied in a local directory.