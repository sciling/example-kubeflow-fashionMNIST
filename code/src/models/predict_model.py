import typing

import kfp.components as comp


def test(
    data_path: comp.InputPath(),
    results_path: comp.OutputPath(),
    labels_dir: comp.OutputPath(),
) -> typing.NamedTuple("Outputs", [("mlpipeline_metrics", "Metrics")]):
    import json
    import os

    import numpy as np
    import tensorflow as tf
    from tensorflow.python import keras

    # Download the dataset and split into training and test data.
    fashion_mnist = keras.datasets.fashion_mnist

    _, (test_images, test_labels) = fashion_mnist.load_data()

    test_images = test_images / 255.0

    # Load the saved Keras model
    model = keras.models.load_model(f"{data_path}/mnist_model.h5")

    # Evaluate the model and print the results
    _, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Test accuracy:", test_acc)

    # Define the class names.
    class_names = [
        "Top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Define a Softmax layer to define outputs as probabilities
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # Classify all the images from the test dataset
    pred_labels = [0 for k in test_images]
    for image_number in range(len(test_images)):
        # Grab an image from the test dataset.
        img = test_images[image_number]

        # Add the image to a batch where it is the only member.
        img = np.expand_dims(img, 0)

        # Predict the label of the image.
        predictions = probability_model.predict(img)

        # Take the prediction with the highest probability
        prediction = np.argmax(predictions[0])
        pred_labels[image_number] = prediction

        # Retrieve the true label of the image from the test labels.
        true_label = test_labels[image_number]

        class_prediction = class_names[prediction]
        confidence = 100 * np.max(predictions)
        actual = class_names[true_label]

        # Save results
        with open(results_path, "a+") as result:
            result.write(
                " Image #:"
                + str(image_number)
                + " | Prediction: {} | Confidence: {:2.0f}% | Actual: {}\n".format(
                    class_prediction, confidence, actual
                )
            )

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Save true labels and predicted labels and class names for confusion matrix
    with open(f"{labels_dir}/true_labels.txt", "w") as ft:
        ft.write(str(list(test_labels)))

    with open(f"{labels_dir}/pred_labels.txt", "w") as fp:
        fp.write(str(pred_labels))

    with open(f"{labels_dir}/class_names.txt", "w") as fp:
        fp.write(str(class_names))

    # Save metrics
    metrics = {
        "metrics": [
            {
                "name": "accuracy",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": str(
                    test_acc
                ),  # The value of the metric. Must be a numeric value.
                "format": "RAW",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            }
        ]
    }
    print("Prediction has been saved successfully!")

    return [json.dumps(metrics)]
