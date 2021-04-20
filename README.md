# unCAPTCHA: AI Recognition of distorted text in images

(v 1.0) | AI, Web Security

---

### Info

**unCAPTCHA** is a simple ML algorithm used to read and write distorted images from FakeCAPTCHA.com.

As of now, the recognition rate of said ML algorithm is about a 97% recognition rate. Being much more faster and considerably better recognition by using data from FakeCAPTCHA.com.

Though, the ML algorithm can *only* recognize digits and not other characters.

[Final Report](./freport.pdf)

---

### Installation

Prerequisites:
- Python 3.x
  - NumPy 1.19.x
  - cv2 4.5.x
  - Matplotlib 3.4.x
  - Tensorflow 2.4.x with Keras
- Anaconda 4.9 

**Activating Conda Virtual Environment**

Before cloning this repository, we advise you to create a separate virtual environment for the ML algorithm to run properly, of course, if you know what you are doing, skip to `Clone the Repo and Running the Program`. This can be done using Anaconda.

```sh
$ conda create -n tf tensorflow
$ conda activate tf
```

This is will help not create any internal issues with non-supercomputers. Now install all the listed pip3 packages needed to run the algorithm, assuming NumPy and TensorFlow are already installed within the virtual environment.

```sh
(tf) $ pip3 install opencv-python
(tf) $ pip3 install matplotlib
```

**Cloning the Repo and Running the Program**

```sh
$ git clone https://github.com/tardiobscurus/unCAPTCHA.git
...
$ cd unCAPTCHA
```

Then run `uncaptcha.py` with a generated CAPTCHA, in this case, we will add `fc001.jpg` (must be either jpg or png)

```sh
$ python3 uncaptcha.py fc001.jpg
```

---


