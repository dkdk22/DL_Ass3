import gc
import json
import urllib.request
import zipfile
from glob import glob, iglob
from os.path import basename, exists

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Input
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.preprocessing import image


def clean_memory():
    gc.collect()
    K.clear_session()


def preprocess_images(
    path, classes=None, preprocessor=None, ext=".jpg", **load_img_kwargs
):
    if classes is None:
        classes = ["."]
    X, y = [], []
    for classidx, classname in enumerate(classes):
        files = list(iglob(f"{path}/{classname}/*{ext}"))
        for img_idx, img_path in enumerate(files):
            print(f"{path}/{classname}: {img_idx} / {len(files)}", end="\r")
            img = image.load_img(img_path, **load_img_kwargs)
            x = image.img_to_array(img)
            img.close()
            if preprocessor:
                x = preprocessor(x)
            X.append(x)
            y.append(classidx)
        print()
    return np.array(X), np.array(y)


def download_image(url, convert_to_array=True, **load_img_kwargs):
    dest = download(url, overwrite=True)
    img = image.load_img(dest, **load_img_kwargs)
    return image.img_to_array(img) if convert_to_array else img


def download(url, dest_filename=None, overwrite=False):
    if not dest_filename:
        dest_filename = basename(url)
    if not overwrite and exists(dest_filename):
        print(
            f"'{dest_filename}' already exists, not overwriting (set overwrite=True to override)"
        )
        return
    print(f"Downloading to '{dest_filename}'...")
    urllib.request.urlretrieve(url, dest_filename)
    return dest_filename


def unzip(source_filename, dest_dir="."):
    print(f"Extracting '{source_filename}' to '{dest_dir}'...")
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)


def get_model_weights(model):
    return {layer.name: layer.get_weights() for layer in model.layers}


def clone_and_set_weights(model, weights, input_tensors):
    new_model = clone_model(model, input_tensors=input_tensors)
    for layer in new_model.layers:
        layer.set_weights(weights[layer.name])
    return new_model


def fit_and_save(model, model_path=None, *fit_args, **fit_kwargs):
    if "learning_rate" in fit_kwargs:
        model.optimizer.lr = fit_kwargs["learning_rate"]
        del fit_kwargs["learning_rate"]
    history = model.fit(*fit_args, **fit_kwargs)
    if model_path:
        model.save(model_path)
    return history


def load_or_build(
    callable,
    model_path,
    *fit_and_save_args,
    **fit_and_save_kwargs,
):
    also_fit_if_not_loaded = True
    if "also_fit_if_not_loaded" in fit_and_save_kwargs:
        also_fit_if_not_loaded = fit_and_save_kwargs["also_fit_if_not_loaded"]
        del fit_and_save_kwargs["also_fit_if_not_loaded"]
    try:
        return load_model(model_path), True
    except Exception:
        pass
    model = callable()
    if also_fit_if_not_loaded:
        fit_and_save(model, model_path, *fit_and_save_args, **fit_and_save_kwargs)
    return model, False


def plot_history(history):
    history_metrics = list(history.history.keys())
    num_plots = len([name for name in history_metrics if not name.startswith("val_")])
    plt_index = 1

    plt.figure(figsize=(10, 3))
    plt.subplot(1, num_plots, plt_index)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.plot(history.epoch, np.array(history.history["loss"]), label="Train Loss")
    if "val_loss" in history_metrics:
        plt.plot(
            history.epoch,
            np.array(history.history["val_loss"]),
            label="Validation Loss",
        )
    plt.legend()
    plt_index += 1

    for name in history_metrics:
        if name == "loss" or name.startswith("val_"):
            continue
        plt.subplot(1, num_plots, plt_index)
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.plot(history.epoch, np.array(history.history[name]), label=f"Train {name}")
        if f"val_{name}" in history_metrics:
            plt.plot(
                history.epoch,
                np.array(history.history[f"val_{name}"]),
                label=f"Validation {name}",
            )
        plt.legend()
        plt_index += 1

    plt.show()


def merge_history(file_pattern_or_list):
    merged_history = {}
    if isinstance(file_pattern_or_list, str):
        file_names = sorted(glob(file_pattern_or_list))
        epochs = len(file_names)
        for file_name in file_names:
            history = json.load(open(file_name, "r"))
            for metric, vals in history.items():
                if metric not in merged_history:
                    merged_history[metric] = []
                merged_history[metric].extend(vals)
    else:
        epochs = len(file_pattern_or_list)
        for history in file_pattern_or_list:
            for metric, vals in history.items():
                if metric not in merged_history:
                    merged_history[metric] = []
                merged_history[metric].extend(vals)
    history = History()
    history.history = merged_history
    history.epoch = list(range(1, epochs + 1))
    return history


class KerasFunctionOptimizer(object):
    def __init__(self, input_shape, disable_eager=True):
        from tensorflow.python.framework.ops import disable_eager_execution

        if disable_eager:
            disable_eager_execution()
        self.input_shape = input_shape
        self.input_tensor = Input(shape=input_shape, name="input")

    def configure_losses(self, losses):
        # Allow the user to configure the losses and define the Keras function
        # We can't do this in init as the user might want to use the input_tensor
        self.loss = K.variable(0.0)
        for loss in losses:
            self.loss = self.loss + loss
        # Calculate the gradients: loss w.r.t. the combination image
        self.gradients = K.gradients(self.loss, self.input_tensor)
        self.outputs = [self.loss, self.gradients]
        self.function = K.function([self.input_tensor], self.outputs)

    def _evaluate(self, x):
        # fmin_l_bfgs_b gave us an 1d array, so we reshape x back to format of our image
        # We also add the batch size in front
        x = x.reshape((1,) + self.input_shape)
        outputs = self.function([x])
        return outputs[0], np.array(outputs[1:]).flatten().astype("float64")

    def _get_loss(self, x):
        self.last_loss, self.last_gradients = self._evaluate(x)
        return self.last_loss

    def _get_gradients(self, x):
        return np.copy(self.last_gradients)

    def optimize(self, x, **args):
        # fmin_l_bfgs_b expects a 1d array, so we flatten x
        x, min_val, info = fmin_l_bfgs_b(
            self._get_loss, x.flatten(), fprime=self._get_gradients, **args
        )
        return x, min_val, info
