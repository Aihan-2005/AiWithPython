# توضیح آموزشی پروژه — FashionMLP Lab

این فایل توضیح می‌دهد **چرا هر قسمت پروژه وجود دارد** و دقیقاً به کدام مفهوم فصل مربوط است.

## 1. مسئله

ورودی هر نمونه یک تصویر خاکستری 28×28 است:

```text
X_i ∈ R^(28×28)
```

کلاس هدف یکی از 10 کلاس Fashion-MNIST است:

```text
y_i ∈ {0, 1, ..., 9}
```

این یک مسئله **Multiclass Classification با کلاس‌های mutually exclusive** است.

بنابراین خروجی مدل 10 Logit دارد و Softmax آن‌ها را به یک توزیع احتمال تبدیل می‌کند:

```text
p_k = exp(z_k) / Σ_j exp(z_j)
```

و:

```text
Σ_k p_k = 1
```

چون Labelها Sparse هستند (مثلاً `y=7`، نه One-Hot)، Loss مناسب:

```text
Sparse Categorical Cross-Entropy
```

است.

## 2. چرا پیکسل‌ها Scale می‌شوند؟

داده خام `uint8` و بین 0 تا 255 است. در `data.py` به float و بازه 0 تا 1 تبدیل می‌شود:

```text
x_scaled = x / 255
```

Gradient-based optimization روی ورودی با مقیاس مناسب پایدارتر است.

## 3. Sequential Model

مدل پایه ساده‌ترین نمایش MLP است:

```text
X → Flatten → Dense/ReLU → Dense/ReLU → Softmax
```

برای Batch با اندازه `m`:

```text
X: m×28×28
Flatten(X): m×784
```

اگر Hidden Layer اول 300 نورون داشته باشد:

```text
W1: 784×300
b1: 300
H1 = ReLU(X_flat W1 + b1)
```

و برای خروجی 10کلاسه:

```text
W_out: n_hidden×10
b_out: 10
P = Softmax(H W_out + b_out)
```

## 4. Backpropagation در پروژه کجاست؟

ما Backprop را دستی نمی‌نویسیم. وقتی `model.fit()` اجرا می‌شود:

```text
Mini-batch
  ↓
Forward pass
  ↓
Cross-entropy loss
  ↓
Reverse-mode automatic differentiation
  ↓
Gradients for every trainable parameter
  ↓
Optimizer update (Adam/SGD)
```

برای SGD، ایده پایه:

```text
θ ← θ - η ∇θ L
```

است.

## 5. Functional Wide & Deep

در `build_functional_classifier()` تصویر Flatten می‌شود. یک مسیر از چند Hidden Layer عبور می‌کند، اما بردار Flat اولیه نیز مستقیماً به بخش انتهایی می‌رود:

```text
flat ─────────────────────────┐
                              ├→ concat → output
flat → hidden1 → hidden2 ─────┘
```

علت آموزشی: نشان دادن این است که Functional API می‌تواند Graphهایی بسازد که Sequential نیستند.

علت مفهومی: خروجی می‌تواند هم از Featureهای مستقیم و هم Featureهای عمیق استفاده کند.

## 6. Auxiliary Output

در مدل `functional_aux` یک خروجی دوم روی Deep Representation قرار می‌گیرد:

```text
hidden_last → aux_output
```

Loss کل:

```text
L = 0.85 L_main + 0.15 L_aux
```

گرادیان لایه‌های Deep از هر دو Loss تأثیر می‌گیرد. این یک نمونه ساده از Multi-output learning و Auxiliary regularization است.

## 7. Subclassing API

کلاس `DynamicMLP(keras.Model)` دو قسمت مهم دارد:

### `__init__()`

مشخص می‌کند مدل **چه اجزایی دارد**:

```text
Flatten
Dense layers
Dropout layers
Output layer
```

### `call()`

مشخص می‌کند اجزا **چطور اجرا می‌شوند**.

Hidden Layerها داخل حلقه اجرا می‌شوند. این رفتار Imperative تفاوت اصلی Subclassing با Functional/Sequential است.

## 8. Learning Rate Finder

LR Finder با LR بسیار کوچک شروع می‌کند:

```text
η_0 = 10^-5
```

و در هر Batch آن را در یک ضریب ثابت ضرب می‌کند:

```text
η_(t+1) = η_t × r
```

تا به LR بزرگ برسد.

در ابتدا LR بسیار کوچک است و Loss آهسته کم می‌شود. در ناحیه مناسب Loss سریع‌تر کاهش می‌یابد. در LR بسیار بزرگ Optimization ناپایدار می‌شود و Loss بالا می‌رود.

هدف، انتخاب LR در ناحیه مناسب قبل از Divergence است.

## 9. Batch Size

اگر Training Set دارای N نمونه و Batch Size برابر B باشد، تعداد Updateهای تقریبی هر Epoch:

```text
ceil(N / B)
```

است.

Batch کوچک Gradient نویزی‌تر دارد. Batch بزرگ Throughput سخت‌افزاری بالاتری می‌دهد ولی رفتار Optimization و Generalization آن می‌تواند متفاوت باشد. به همین دلیل در Tuner، Batch Size نیز Tune می‌شود.

## 10. EarlyStopping

به‌جای حدس زدن اینکه 20 یا 50 یا 100 Epoch مناسب است، Epoch سقف بزرگی در نظر می‌گیریم و `EarlyStopping` Validation Loss را Monitor می‌کند.

اگر برای چند Epoch بهبود نداشت:

```text
stop_training = True
```

و با `restore_best_weights=True` Weightهای بهترین Epoch بازیابی می‌شوند.

## 11. ModelCheckpoint

بهترین مدل روی دیسک ذخیره می‌شود. تفاوت آن با EarlyStopping:

- EarlyStopping: تصمیم به توقف Training
- ModelCheckpoint: ذخیره State خوب روی Disk

است.

## 12. Custom Callback

`ValTrainRatioCallback` در `on_epoch_end()` این نسبت را محاسبه می‌کند:

```text
val_loss / train_loss
```

اگر فاصله Validation و Train با گذشت زمان زیاد شود، می‌تواند علامتی از Overfitting باشد. این نسبت فقط یک Diagnostic است و قانون قطعی نیست.

## 13. TensorBoard

هر Run یک Log Directory مستقل دارد. بنابراین می‌توان مثلاً این دو آزمایش را مقایسه کرد:

```text
Adam, lr=1e-3, batch=128
SGD,  lr=5e-2, batch=64
```

و Learning Curves را روی یک UI دید.

## 14. Hyperparameter Tuning

Search Space پروژه:

```text
n_hidden        ∈ {1,2,3,4,5}
units           ∈ {64,128,...,512}
dropout         ∈ {0.0,0.1,...,0.5}
optimizer       ∈ {adam, sgd}
learning_rate   ∈ log-uniform range
batch_size      ∈ {32,64,128,256}
```

Learning Rate با sampling لگاریتمی جستجو می‌شود چون تفاوت مقیاسی بین `1e-4` و `1e-3` مهم‌تر از فاصله خطی ساده است.

### Random Search

ترکیب‌ها را تصادفی انتخاب می‌کند.

### Hyperband

Trialهای زیادی را با Resource کم شروع می‌کند و مدل‌های ضعیف را زود حذف می‌کند تا Resource بیشتری به مدل‌های خوب برسد.

### Bayesian Optimization

از نتایج Trialهای قبلی برای تخمین ناحیه‌های امیدوارکننده Search Space استفاده می‌کند.

## 15. چرا Validation و Test جدا هستند؟

Hyperparameterها براساس Validation انتخاب می‌شوند. Test Set باید تا انتخاب نهایی مدل دست‌نخورده بماند.

اگر مرتب Test Accuracy را ببینیم و معماری را براساس آن عوض کنیم، Test Set عملاً وارد فرایند Model Selection شده و تخمین Generalization خوش‌بینانه می‌شود.

## 16. Evaluation

`evaluate.py` این موارد را می‌سازد:

- Test loss/accuracy
- Classification report
- Confusion matrix
- چند نمونه Prediction

Confusion Matrix نشان می‌دهد مدل کدام کلاس‌ها را با هم اشتباه می‌گیرد. در Fashion-MNIST معمولاً کلاس‌هایی مانند Shirt / T-shirt / Pullover از نظر بصری شباهت بیشتری دارند و بررسی Confusionها ارزشمند است.

## 17. چیزی که در مصاحبه باید بتوانی توضیح دهی

اگر از این پروژه سؤال شد، فقط APIها را نام نبر. این زنجیره را توضیح بده:

```text
I started with a reproducible MLP baseline,
validated the optimization regime using an LR range test,
added early stopping/checkpointing and TensorBoard,
then tuned architecture + optimizer + learning rate + batch size
without touching the test set,
and finally evaluated the selected model with class-level diagnostics.
```

این نشان می‌دهد فقط مدل ننوشته‌ای؛ **Experiment Design و Model Selection** را هم می‌فهمی.
