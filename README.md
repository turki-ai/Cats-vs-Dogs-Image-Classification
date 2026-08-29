# Cats vs Dogs Image Classification

Binary image classification pipeline using transfer learning with a pre-trained MobileNetV2 backbone in TensorFlow/Keras.

---

## Overview

* Raw archive extraction and automated cleanup of non-JPEG artifacts.
* Data augmentation (rotation, zoom, width/height shifts, horizontal flips).
* Pre-trained MobileNetV2 feature extraction layer (ImageNet weights, frozen).
* Binary cross-entropy optimization with `EarlyStopping`.
* Checkpointed model serialization to `.keras` format.
* Single-image inference and visual prediction rendering.

---

## Dataset Structure

The dataset contains ~10,000 images split into training and test directories:

```text
dataset/
├── training_set/
│   └── training_set/
│       ├── cats/  [4,001 images]
│       └── dogs/  [4,006 images]
└── test_set/
    └── test_set/
        ├── cats/  [1,012 images]
        └── dogs/  [1,013 images]
```

---

## Technical Specifications

| Parameter | Value |
| :--- | :--- |
| **Input Shape** | 224 x 224 x 3 |
| **Base Model** | MobileNetV2 (Feature Extractor) |
| **Output Layer** | Dense (1 unit, Sigmoid) |
| **Loss Function** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Batch Size** | 64 |
| **Final Accuracy** | ~98.5% |
| **Final Loss** | ~0.038 |
| **Saved Format** | `.keras` |

---

## Implementation Details

### Data Preprocessing & Augmentation

```python
trainDataAugmented = ImageDataGenerator(
    rescale=1/255,
    rotation_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)
testDataAugmented = ImageDataGenerator(rescale=1/255)

trainDataAugmentedShuffled = trainDataAugmented.flow_from_directory(
    trainDir,
    target_size=(224, 224),
    shuffle=True,
    class_mode="binary",
    batch_size=64
)

testDataAugmentedShuffled = testDataAugmented.flow_from_directory(
    testDir,
    target_size=(224, 224),
    class_mode="binary",
    batch_size=64
)
```

### Architecture & Training

```python
url = "[https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4](https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4)"

computerVisionCatVSDogs = Sequential([
    Lambda(lambda x: hub.KerasLayer(url, trainable=False)(x), input_shape=(224, 224, 3)),
    Dense(1, activation='sigmoid')
])

computerVisionCatVSDogs.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = computerVisionCatVSDogs.fit(
    trainDataAugmentedShuffled,
    epochs=100,
    validation_data=testDataAugmentedShuffled,
    validation_batch_size=len(testDataAugmentedShuffled),
    callbacks=[early_stop]
)

computerVisionCatVSDogs.save('model.keras')
```

### Inference Pipeline

```python
def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.0
    return img

def pred_and_plot(model, filename, class_names=classNames):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[int(tf.round(pred))]

    catOrDog = 'A Cat' if pred_class == 'cats' else 'A Dog'

    plt.imshow(img)
    plt.title(f"Prediction: {catOrDog}")
    plt.axis(False)
    plt.show()
```

---

## Environment & Stack

* Python 
* TensorFlow 
* TensorFlow Hub
* NumPy
* Matplotlib
