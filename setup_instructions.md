
# ALPR Environment Setup

This guide will walk you through setting up a Python 2.7 virtual environment using `pyenv`, and installing the necessary libraries for the ALPR (Automatic License Plate Recognition) project. We will use specific versions of TensorFlow, Keras, and OpenCV for compatibility with older codebases.

## Prerequisites

Make sure you have the following installed:

- `zsh` or `bash` shell
- Git
- curl

## Step 1: Install or Reconfigure `pyenv`

### 1.1 Check if `pyenv` is installed:

Run the following command to check if `pyenv` is installed:

```bash
pyenv --version
```

If you get `command not found`, proceed with the installation steps below.

### 1.2 Remove any existing `pyenv` installation:

If you've previously installed `pyenv` and want to start fresh, run:

```bash
rm -rf ~/.pyenv
```

### 1.3 Install `pyenv`:

Run the following command to install `pyenv`:

```bash
curl https://pyenv.run | bash
```

### 1.4 Add `pyenv` to your shell profile:

Open your `~/.zshrc` (if you're using `zsh`) or `~/.bashrc` (for bash) and add the following lines to load `pyenv`:

```bash
# Add pyenv to PATH
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Reload the shell configuration to apply the changes:

```bash
source ~/.zshrc   # For zsh users
# or
source ~/.bashrc  # For bash users
```

### 1.5 Verify `pyenv` installation:

Check if `pyenv` is working:

```bash
pyenv --version
```

## Step 2: Set Up the Python 2.7 Virtual Environment

### 2.1 Install Python 2.7:

Install Python 2.7.18 using `pyenv`:

```bash
pyenv install 2.7.18
```

### 2.2 Create a Virtual Environment:

Create a new virtual environment named `alpr-env-py27` using Python 2.7.18:

```bash
pyenv virtualenv 2.7.18 alpr-env-py27
```

### 2.3 Activate the Virtual Environment:

Activate the newly created virtual environment:

```bash
pyenv activate alpr-env-py27
```

You should see the environment name in your terminal prompt: `(alpr-env-py27)`.

## Step 3: Install Required Libraries

### 3.1 Install TensorFlow 1.5.0:

Install TensorFlow 1.5.0 for Python 2.7:

```bash
pip install --no-cache-dir https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.5.0-cp27-cp27m-linux_x86_64.whl
```

### 3.2 Install Keras 2.2.4:

```bash
pip install keras==2.2.4
```

### 3.3 Install OpenCV:

Install OpenCV version 3.4.2.17:

```bash
pip install opencv-python==3.4.2.17
```

## Step 4: Verify Installation

You can verify the installation by importing the libraries in a Python shell:

```bash
python
>>> import tensorflow as tf
>>> import keras
>>> import cv2
>>> print(tf.__version__)  # Should print 1.5.0
>>> print(keras.__version__)  # Should print 2.2.4
>>> print(cv2.__version__)  # Should print 3.4.2
```

