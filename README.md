<div align="center">

# Plant Disease Detection

**Identify plant leaf diseases with a PyTorch CNN, explore treatment information, and get seasonal crop suggestions.**

<img src="Plant-Disease-Detection-main/demo_images/1.png" alt="Plant disease detection app home screen" width="900">

</div>

## What it does

- Classifies leaf images across 39 plant disease categories.
- Shows disease details, prevention guidance, and supplement information.
- Recommends crops based on region and season.
- Includes sample images, model training material, and a Flask web interface.

## Screenshots

<table>
  <tr>
    <td align="center"><strong>Leaf diagnosis</strong><br><img src="Plant-Disease-Detection-main/demo_images/2.png" alt="Leaf diagnosis page" width="480"></td>
    <td align="center"><strong>Diagnosis results</strong><br><img src="Plant-Disease-Detection-main/demo_images/3.png" alt="Disease diagnosis results" width="480"></td>
  </tr>
  <tr>
    <td align="center"><strong>Plant care marketplace</strong><br><img src="Plant-Disease-Detection-main/demo_images/4.JPG" alt="Plant care marketplace" width="480"></td>
    <td align="center"><strong>Additional app view</strong><br><img src="Plant-Disease-Detection-main/demo_images/5.png" alt="Additional app screen" width="480"></td>
  </tr>
</table>

## Run locally

Requires Python 3.11. The model and supporting data files are included in the repository.

```powershell
cd "Plant-Disease-Detection-main/Flask Deployed App"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_updated.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Project layout

| Path | Contents |
| --- | --- |
| `Plant-Disease-Detection-main/Flask Deployed App/` | Flask app, CNN, model files, templates, and static assets |
| `Plant-Disease-Detection-main/demo_images/` | Application screenshots |
| `Plant-Disease-Detection-main/test_images/` | Example leaf images for testing |
| `Plant-Disease-Detection-main/Model/` | Training notebook and model documentation |
| `render.yaml` | Render web service configuration |

The 210 MB PyTorch checkpoint is tracked with Git LFS. Install Git LFS before cloning if you need to run inference locally.