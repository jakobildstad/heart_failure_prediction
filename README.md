# "Heart failure detector" (coronary heart disease)
## Learning project
- with help from ChatGPT
- dataset from: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download
---
## Quickstart
- $ *python main.py train --csv data/heart.csv*
    - train the model on given data
- $ *python main.py evaluate -m model/best_heartnet.pth*
    - evaluate the models performance with metrcis like accuracy, EER, etc.
- $ *python main.py predict -m model/best_heartnet.pth*
    - Get a score by manually choosing values for the different inputs.