# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Building the Kubeflow Pipeline
# 
# Up until this point, all our steps are similar to what you can find in the [Basic classification with Tensorflow](https://https://www.tensorflow.org/tutorials/keras/classification) example. The next step on that example is to build the model. However, we will make use of the containerized approach provided by Kubeflow to allow our model to be run using Kubernetes.
# %% [markdown]
# ### 2.1 Install Kubeflow pipelines SDK
# %% [markdown]
#  The first step is to install the Kubeflow Pipelines SDK package.
# 
# 

# %% [markdown]
# ### 2.2 Build Container Components
# %% [markdown]
# The following cells define functions that will be transformed into lightweight container components. It is recommended to look at the corresponding Fashion MNIST notebook to match what you see here to the original code.

# %%
# Import Kubeflow SDK
import kfp
import kfp.dsl as dsl
import kfp.components as comp


# %%
def train(data_path, model_file):
    
    # func_to_container_op requires packages to be imported inside of the function.
    import pickle
    
    import tensorflow as tf
    from tensorflow.python import keras
    
    # Download the dataset and split into training and test data. 
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the data so that the values all fall between 0 and 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model using Keras.
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Run a training job with specified number of epochs
    model.fit(train_images, train_labels, epochs=10)

    # Evaluate the model and print the results
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('Test accuracy:', test_acc)

    # Save the model to the designated 
    model.save(f'{data_path}/{model_file}')

    # Save the test_data as a pickle file to be used by the predict component.
    with open(f'{data_path}/test_data', 'wb') as f:
        pickle.dump((test_images,test_labels), f)


# %%
def predict(data_path, model_file, image_number):
    
    # func_to_container_op requires packages to be imported inside of the function.
    import pickle

    import tensorflow as tf
    from tensorflow import keras

    import numpy as np
    
    # Load the saved Keras model
    model = keras.models.load_model(f'{data_path}/{model_file}')

    # Load and unpack the test_data
    with open(f'{data_path}/test_data','rb') as f:
        test_data = pickle.load(f)
    # Separate the test_images from the test_labels.
    test_images, test_labels = test_data
    # Define the class names.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Define a Softmax layer to define outputs as probabilities
    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])

    # See https://github.com/kubeflow/pipelines/issues/2320 for explanation on this line.
    image_number = int(image_number)

    # Grab an image from the test dataset.
    img = test_images[image_number]

    # Add the image to a batch where it is the only member.
    img = (np.expand_dims(img,0))

    # Predict the label of the image.
    predictions = probability_model.predict(img)

    # Take the prediction with the highest probability
    prediction = np.argmax(predictions[0])

    # Retrieve the true label of the image from the test labels.
    true_label = test_labels[image_number]
    
    class_prediction = class_names[prediction]
    confidence = 100*np.max(predictions)
    actual = class_names[true_label]
    
    
    with open(f'{data_path}/result.txt', 'w') as result:
        result.write(" Prediction: {} | Confidence: {:2.0f}% | Actual: {}".format(class_prediction,
                                                                        confidence,
                                                                        actual))
    
    print('Prediction has be saved successfully!')


# %%
# Create train and predict lightweight components.
train_op = comp.func_to_container_op(train, base_image='tensorflow/tensorflow:latest-gpu-py3')
predict_op = comp.func_to_container_op(predict, base_image='tensorflow/tensorflow:latest-gpu-py3')

# %% [markdown]
# ### 2.3 Build Kubeflow Pipeline
# %% [markdown]
# Our next step will be to create the various components that will make up the pipeline. Define the pipeline using the *@dsl.pipeline* decorator.
# 
# The pipeline function is defined and includes a number of paramters that will be fed into our various components throughout execution. Kubeflow Pipelines are created decalaratively. This means that the code is not run until the pipeline is compiled. 
# 
# A [Persistent Volume Claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) can be quickly created using the [VolumeOp](https://) method to save and persist data between the components. Note that while this is a great method to use locally, you could also use a cloud bucket for your persistent storage.

# %%
# Define the pipeline
@dsl.pipeline(
   name='MNIST Pipeline',
   description='A toy pipeline that performs mnist model training and prediction.'
)

# Define parameters to be fed into pipeline
def mnist_container_pipeline(
    data_path: str,
    model_file: str, 
    image_number: int
):
    
    # Define volume to share data between components.
    vop = dsl.VolumeOp(
    name="create_volume",
    resource_name="data-volume", 
    size="1Gi", 
    modes=dsl.VOLUME_MODE_RWO)
    
    # Create MNIST training component.
    mnist_training_container = train_op(data_path, model_file)                                     .add_pvolumes({data_path: vop.volume})

    # Create MNIST prediction component.
    mnist_predict_container = predict_op(data_path, model_file, image_number)                                     .add_pvolumes({data_path: mnist_training_container.pvolume})
    
    # Print the result of the prediction
    mnist_result_container = dsl.ContainerOp(
        name="print_prediction",
        image='library/bash:4.4.23',
        pvolumes={data_path: mnist_predict_container.pvolume},
        arguments=['cat', f'{data_path}/result.txt']
    )