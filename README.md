# ✋ International Hand Sign Detection

A real-time hand sign recognition system using a webcam, OpenCV, and a deep learning model trained to recognize international hand sign letters (A–Z and space).


<p align="center">
  <img src="assets/sign_language_chart.jpg" alt="1_data_profession_table" width="1000">
</p>

---

### 📂 Project Structure

```
├── model/
│   ├── hand_sign_detection_model.h5      # Trained classification model
│   ├── labels.txt                         # Label file mapping output indexes to characters
├── main.py                                # Main script for detection
├── text.txt                               # Output file storing translated sentences
├── README.md                              # Project documentation
```

---

### 🛠 Requirements

* Python 3.8+
* OpenCV (`cv2`)
* NumPy
* cvzone
* TensorFlow / Keras

You can install dependencies with:

```bash
pip install opencv-python numpy cvzone tensorflow
```

---

### 🚀 How to Run

1. Make sure your webcam is connected.
2. Place your hand in front of the camera forming sign language letters.
3. Run the script:

```bash
python main.py
```

---

### ⌨️ Controls

* Press **`r`** → Reset the sentence.
* Press **`s`** → Save the sentence to `text.txt`.
* Press **`ESC`** → Exit the application.

---

### 🧠 Logic & Features

* The webcam captures your hand using `cvzone.HandTrackingModule`.
* The hand region is cropped and resized into a square input (300x300).
* A pre-trained model (`hand_sign_detection_model.h5`) predicts the letter.
* Space between words is detected via the underscore (`_`) class.
* Debouncing is used to avoid repeating characters when holding a sign too long:

  * Delay between different characters.
  * Timeout to auto-append the same character if held long enough.

---

### 📖 Training

This model assumes a CNN trained on hand gesture images with corresponding labels in `labels.txt`. Each label should match a letter A–Z or `_` for space.

---

### 📸 Example Output

If you sign the letters `H-E-L-L-O`, the text shown on the screen will be:

```
HELLO
```

Saving it (`s`) will write it into `text.txt`.

---




