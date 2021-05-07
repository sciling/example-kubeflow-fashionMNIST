import typing

import kfp.components as comp


def train(
    compile_optimizer,
    epochs,
    validation_split,
    data_path: comp.OutputPath(),
    lossplot_path: comp.OutputPath(str),
) -> typing.NamedTuple("loss_plot", [("mlpipeline_ui_metadata", "UI_metadata")]):

    import base64
    import json
    import os
    from collections import namedtuple

    import matplotlib.pyplot as plt
    from tensorflow.python import keras

    # Parse pipeline parameters
    epochs = int(epochs)
    validation_split = float(validation_split)

    def save_loss_plot(history, plot_path):
        """
        history: History object from keras. Its History.history attribute is a record of training loss and validation loss values.
        plot_path: path where plot image will be saved.
        """
        # Creation of the plot
        loss, val_loss = history.history["loss"], history.history["val_loss"]
        fig = plt.figure(figsize=(30, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

        # Saving plot in specified path
        with open(plot_path, "wb") as fd:
            plt.savefig(fd)

    def get_web_app_from_loss_plot(plot_path):
        """
        plot_path: path where plot image is saved.
        return: JSON object representing kubeflow output viewer for web-app.
        """
        # Retrieve encoded bytes of the specified image path
        with open(plot_path, "rb") as fd:
            encoded = base64.b64encode(fd.read()).decode("latin1")

        web_app_json = {
            "type": "web-app",
            "storage": "inline",
            "source": f"""<img width="100%" src="data:image/png;base64,{encoded}"/>""",
        }
        return web_app_json

    # Download the dataset and split into training and test data.
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), _ = fashion_mnist.load_data()

    # Normalize the data so that the values all fall between 0 and 1.
    train_images = train_images / 255.0

    # Define the model using Keras.
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=compile_optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Run a training job with specified number of epochs
    history = model.fit(
        train_images, train_labels, epochs=epochs, validation_split=validation_split
    )

    # Save loss plot
    save_loss_plot(history, lossplot_path)

    loss_plot = [get_web_app_from_loss_plot(lossplot_path)]

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Save the model to the specified output path
    model.save(f"{data_path}/mnist_model.h5")

    print("============== END TRAINING ==============")

    # Return specified loss_plot
    metadata = {"outputs": loss_plot}

    loss_plot = namedtuple("loss_plot", ["mlpipeline_ui_metadata"])
    return loss_plot(json.dumps(metadata))
