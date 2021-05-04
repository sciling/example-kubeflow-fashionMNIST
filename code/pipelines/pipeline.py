import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))


import kfp
import kfp.components as comp
import kfp.dsl as dsl


# Define the pipeline
@dsl.pipeline(
    name="MNIST Pipeline",
    description="A toy pipeline that performs mnist model training and prediction.",
)
# Define parameters to be fed into pipeline
def mnist_container_pipeline(
    compile_optimizer: str = "adam",
    epochs: int = 50,
    validation_split: float = 0.15,
):
    from models.predict_model import test
    from models.train_model import train
    from visualization.confusion_matrix import confusion_matrix

    # Create train and predict lightweight components.
    train_op = comp.func_to_container_op(
        train,
        base_image="tensorflow/tensorflow:latest-gpu-py3",
        packages_to_install=["matplotlib"],
    )
    test_op = comp.func_to_container_op(
        test,
        base_image="tensorflow/tensorflow:latest-gpu-py3",
        packages_to_install=["scikit-learn"],
    )
    confusion_matrix_op = comp.func_to_container_op(
        confusion_matrix, packages_to_install=["scikit-learn"]
    )

    # Create MNIST training component.
    mnist_training_container = train_op(compile_optimizer, epochs, validation_split)
    mnist_training_container.execution_options.caching_strategy.max_cache_staleness = (
        "P0D"
    )

    # Create MNIST prediction component.
    mnist_predict_container = test_op(mnist_training_container.outputs["data"])

    confusion_matrix_op(mnist_predict_container.outputs["labels_dir"])


if __name__ == "__main__":

    # Compile pipeline to generate compressed YAML definition of the pipeline.
    kfp.compiler.Compiler().compile(
        mnist_container_pipeline, "{}.zip".format("fashion_mnist_kubeflow")
    )
