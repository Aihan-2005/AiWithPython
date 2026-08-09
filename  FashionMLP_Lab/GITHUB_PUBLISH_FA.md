# چک‌لیست انتشار پروژه در GitHub

## 1. سه Placeholder را عوض کن

در فایل‌های `README.md`، `pyproject.toml` و `LICENSE` موارد زیر را با اطلاعات خودت جایگزین کن:

- `YOUR_GITHUB_REPOSITORY_URL`
- `Your Name`
- در صورت تمایل توضیح کوتاه شخصی در README

## 2. پروژه را نصب کن

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## 3. تست‌ها را اجرا کن

```bash
pytest -q
```

## 4. Baseline را آموزش بده

```bash
fashion-mlp-train --model sequential --hidden-units 300 100 --epochs 50
```

Metric واقعی Test را از `runs/.../metrics.json` بردار. هیچ عددی را حدس نزن.

## 5. LR Finder را اجرا کن

```bash
fashion-mlp-lr-find --start-lr 1e-5 --end-lr 1 --steps 500
```

تصویر `lr_finder.png` را بررسی کن و یک Learning Rate منطقی انتخاب کن.

## 6. حداقل یک Hyperparameter Search اجرا کن

برای شروع Hyperband انتخاب خوبی است:

```bash
fashion-mlp-tune --strategy hyperband --epochs 30 --overwrite
```

## 7. مدل نهایی را Evaluate کن

```bash
fashion-mlp-evaluate runs/<RUN>/best_model.keras
```

از خروجی‌های زیر برای GitHub استفاده کن:

- `confusion_matrix.png`
- `sample_predictions.png`
- `learning_curves_loss.png`
- `learning_curves_metrics.png`

تصاویر منتخب را به `docs/images/` کپی کن تا GitHub آن‌ها را Track کند.

## 8. TensorBoard Screenshot بگیر

```bash
tensorboard --logdir runs
```

دو Run با Learning Rate یا Optimizer متفاوت را مقایسه کن و Screenshot را به `docs/images/` اضافه کن.

## 9. README را با نتیجه واقعی کامل کن

بعد از اجرای پروژه، در README یک بخش Results اضافه کن. نمونه:

```markdown
## Results

| Model | Optimizer | LR | Batch | Test Accuracy |
|---|---|---:|---:|---:|
| Sequential MLP | Adam | ... | ... | ... |
| Functional Wide & Deep | Adam | ... | ... | ... |
| Tuned MLP | ... | ... | ... | ... |
```

فقط Metricهایی را بنویس که واقعاً تولید کرده‌ای.

## 10. Git Repository بساز

```bash
git init
git add .
git commit -m "Build end-to-end Fashion-MNIST MLP experimentation framework"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## 11. GitHub Description پیشنهادی

```text
End-to-end Fashion-MNIST MLP experimentation framework with Keras 3: Sequential/Functional/Subclassing APIs, LR finder, callbacks, TensorBoard, and automated hyperparameter tuning.
```

## 12. Topicهای پیشنهادی GitHub

```text
deep-learning
neural-networks
keras
tensorflow
keras-tuner
fashion-mnist
hyperparameter-tuning
tensorboard
machine-learning
python
```
