# image_classifier

This is a simple **Flask-based image classifier** project. It allows users to upload images, organize them into datasets, and run classification.

---

## 🚀 Features
- Upload images and process them.
- Organized template system (`index.html`, `result.html`, `upload.html`).
- Structured project folders (`data/`, `templates/`, etc.).
- Written in Python for easy customization.

---

## 📂 Project Structure
```
├── data/ # Sample images
│ ├── Darlene/
│ ├── Jake/
│ └── Sunshine/
├── temp/
│ └── templates/ # HTML templates (index, upload, result)
├── venv/ # Virtual environment (ignored in Git)
├── app.py # Main Flask application
├── test.py # Test script
└── .gitignore
```
---

## ⚙️ Installation & Setup

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/darlenelovitos/image_classifier
   python -m venv venv
   ```
2. **Create a virtual environment**

  ```bash
  source venv/bin/activate   # On Mac/Linux
  venv\Scripts\activate      # On Windows
  ```
3. **Install dependencies**

  ```bash
  pip install flask tensorflow pillow numpy scipy
  ```

4. **Run the app locally:**

  ```bash
  python app.py
  ```
  Then open your browser at http://127.0.0.1:5000/

## 📌 Requirements
<ul>
<li>Python 3.11.1</li>
<li>Visual Studio Code</li>
</ul>

## ✅ Usage
<ul>
  <li>Open the app in your browser.</li>
  <li>Add classes matching your dataset.</li>
  <li>Upload images into the data/ folder.</li>
  <li>Train the dataset and classify new images.</li>
</ul>

## 📜 License
This project is licensed under the MIT License – see the LICENSE file for details.


