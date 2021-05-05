# SAME Example: Fashion MNIST Clothing Category Prediction

[![Code Quality](https://github.com/SAME-Project/example-kubeflow-fashionMNIST/actions/workflows/quality.yml/badge.svg)](https://github.com/SAME-Project/example-kubeflow-fashionMNIST/actions/workflows/quality.yaml) [![Tests](https://github.com/SAME-Project/example-kubeflow-fashionMNIST/actions/workflows/test.yml/badge.svg)](https://github.com/SAME-Project/example-kubeflow-fashionMNIST/actions/workflows/test.yml) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

**TL;DR;** Reproducible clothing classification training and evaluation pipelines in a single command.

- [Installing / Getting Started](#installing--getting-started)
  * [Pipeline Parameters](#pipeline-parameters)
  * [Pipeline Stages](#pipeline-stages)
- [Experimental Results](#experimental-results)
  * [Input Parameters](#input-parameters)
  * [Loss Plot](#loss-plot)
  * [Confusion Matrix](#confusion-matrix)
  * [Metrics](#metrics)
- [Developing](#developing)
  * [Testing](#testing)
- [Known Issues](#known-issues)
  * [Tensorboard](#tensorboard)
- [Contributing](#contributing)
- [Credits](#credits)

This project demonstrates a common industrial task of eCommerce classification. You can imagine that this model helps users or operational teams quickly classify a product into a catagory, just from a photo, ready for entering into the product catalogue. It's important to have a repeatable pipeline and scaffold for this project, because multiple teams might be interacting with it.

## Installing / Getting Started

Create a working SAME installation by [following instructions found in the wiki](https://github.com/azure-octo/same-cli/wiki/Epic-Sprint-1-Demo), but stop before the "Run a program" section. Then run the following commands:

```bash
git clone https://github.com/SAME-Project/example-kubeflow-fashionMNIST
cd example-kubeflow-fashionMNIST
same program create -f same.yaml
same program run -f same.yaml --experiment-name example-kubeflow-fashionMNIST --run-name default
```

Now browse to your kubeflow installation and you should be able to see an experiment and a run.

### Pipeline Parameters

| Pipeline parameter | Description |
| ------ | ------ |
|compile_optimizer| String (name of optimizer) or optimizer instance. See tf.keras.optimizers. (e.g "adam")|
|epochs| Integer. Number of epochs to train the model. (e.g 50)|
|validation_split| Float between 0 and 1. Can't be 0. Fraction of the training data to be used as validation data(e.g 0.1)|

### Pipeline Stages

#### 1. Train ([code](./src/train.py))
This component performs the following operations:

    1. Loads the keras dataset
    2. Uses the training set to train a model
    3. Save the model in an OutputPath Artifact.
    4. Generate a loss plot, saves it in an OutputArtifact and embed its visualization inside a web-app component.

#### 2. Test ([code](./src/test.py))
This component performs the following operations:

    1. Loads the previously saved model through an InputPath Artifact.
    2. Uses the testing set to test the model.
    3. Saves the image, prediction and confidence inside a file generated as an OutputPath Artifact(results_path).
    4. Saves true and predicted labels, as well as class names to pass it later     to the confusion matrix.
    5. Generate accuracy as metrics

#### 3. Confusion matrix ([code](./src/confusion_matrix.py))
This component is passed the labels directory(which contains true and predicted labels, as well as class names) and generates a confusion matrix that kubeflow UI can understand.This function can be reused in other pipelines if given the appropiate parameters.

## Experimental Results

In this section we will replicate the results for the fashion experiment.
The pipeline outputs are a confusion matrix ,a loss plot and metrics for the accuracy of the model, from which metrics can be directly compared.

We can see them in the visualizations of the pipeline or in the Run Output Tab of the Run.

In order to check the validity of the pipeline, we are going to execute a run with the same parameters as the original experiment and compare the outputs with the ones obtained in [Basic classification: Classify images of clothing"](https://www.tensorflow.org/tutorials/keras/classification#evaluate_accuracy). As we can't put the validation split parameter to 0 (as we need some values to do the loss plot), we will put a very small value (0.05).

### Input Parameters
| Pipeline parameter | Value |
| ------ | ------ |
|compile_optimizer|adam|
|epochs|10|
|validation_split|0.05|

### Loss Plot

![lossplot.png](./code/data/images/loss_plot.png)

### Confusion Matrix

![confmatrix.png](./code/data/images/confusion_matrix.png)

### Metrics

The original results are shown in https://www.tensorflow.org/tutorials/keras/classification#evaluate_accuracy. In particular, the results for the the fashion task are an accuracy of 0.8693000078201294

In our replication, we get similar results:
![metrics.png](./code/data/images/metrics.png)

If we increase the number of epochs to 100, and the validation split to 0.10, the results remain more or less the same: 0.8819

## Developing

When attempting to run or test the code locally you will need to install the reqiured libraries (this requires [poetry](https://python-poetry.org)).

```bash
make install
```

### Testing

This repo is not a library, nor is it meant to run with different permutations of Python or library versions. It is not guaranteed to work with different Python or library versions, but it might. There is limited matrix testing in the github action CI/CD.

```bash
make tests
```

## Known Issues

### Tensorboard

Waiting on [this PR](https://github.com/kubeflow/pipelines/pull/5515) and subsequent KFP release. Supposed to be [possible with volumes](https://github.com/kubeflow/pipelines/issues/4850), but I had caching issues. Basically worked first time, but never again.

## Contributing

See [CONTRUBUTING.md](CONTRIBUTING.md).

## Credits

This project was delivered by [Winder Research](https://WinderResearch.com), an ML/RL/MLOps consultancy.

This work is based on the github project [From Notebook to Kubeflow Pipeline using Fashion MNIST](https://github.com/manceps/fashion-mnist-kfp-lab/blob/master/KF_Fashion_MNIST.ipynb) under the [MIT License](https://github.com/manceps/fashion-mnist-kfp-lab/blob/master/LICENSE).
