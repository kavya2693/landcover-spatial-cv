# ML Interview Q&A — Land-Cover Classification (EuroSAT)

A beginner-friendly study guide anchored to **this** project: classifying ~27,000 Sentinel-2 satellite image patches (64×64 pixels, 10 land-cover classes) using a **ResNet18** transfer-learning baseline (ImageNet-pretrained, frozen backbone, new final layer), trained on CPU (Apple M1), evaluated with overall accuracy, Cohen's kappa, and a confusion matrix. Later phases add spatial cross-validation and a CNN-vs-transformer benchmark with per-class IoU.

Read each model answer out loud until you can say it in your own words. The goal is understanding, not memorizing.

---

## Section 1 — Dataset & Problem Framing

### Q1.1 — What is this project doing, in one sentence?
It takes a 64×64-pixel satellite image patch and predicts which one of 10 land-cover types it shows (for example Forest, River, Industrial, or AnnualCrop), using the EuroSAT dataset of about 27,000 labeled Sentinel-2 patches. This is a **single-label image classification** problem: every patch gets exactly one of 10 answers. The model learns the visual patterns (texture, color, shape) that separate, say, a forest from an industrial zone. The business framing is automated mapping of how land is being used from satellite imagery.

**Interview tip:** Lead with the input, the output, and the number of classes — interviewers immediately know how to think about your problem.

### Q1.2 — Is this classification or regression, and why?
It is **classification**, because the output is a discrete category (one of 10 land-cover labels), not a continuous number. If we were instead predicting "percent of this patch that is water" we would be doing regression. Because each patch has exactly one correct label, it is *multi-class single-label* classification (10 mutually exclusive classes), not *multi-label* (where a patch could be both "forest" and "river" at once).

### Q1.3 — What does the data actually look like?
Each example is a small color image: 64 pixels wide, 64 pixels tall, with 3 channels (red, green, blue) in the standard RGB version of EuroSAT. So one image is a grid of 64 × 64 × 3 = 12,288 numbers, each a pixel intensity. There are ~27,000 such images spread across 10 classes. The classes are roughly balanced but not perfectly equal, which matters for how we measure success later.

### Q1.4 — Why 64×64? Isn't that tiny?
EuroSAT patches are 64×64 because they come from Sentinel-2 satellite tiles at 10-meter resolution, so one patch covers a real-world area of about 640×640 meters. That is enough to see the *texture* of land use (the regular grid of a crop field, the rough canopy of a forest) even if you cannot see individual objects. Small images are also cheaper to train, which matters here because we train on a CPU. The trade-off: fine details are lost, so visually similar classes can be confused.

### Q1.5 — How did you split the data into train and test sets?
The baseline uses a **random split**, typically something like 70% train / 15% validation / 15% test, with the split stratified so each class keeps its proportion in every part. The training set is what the model learns from, the validation set is used to tune choices and watch for overfitting, and the test set is touched only once at the end to report honest performance. The critical caveat — explored in Section 6 — is that a *random* split can leak information when images are spatially close on the map.

---

## Section 2 — CNNs & ResNet

### Q2.1 — What is a CNN and why use one for images?
A Convolutional Neural Network (CNN) is a model that slides small filters across an image to detect local patterns — edges, then textures, then shapes, then whole objects — building up from simple to complex. It is the natural choice for images because it respects the fact that nearby pixels are related and that a pattern (like a tree texture) means the same thing wherever it appears in the image. This property, called *translation invariance*, means a CNN needs far fewer parameters than treating every pixel independently. For our 64×64 satellite patches, a CNN can learn "what forest texture looks like" once and recognize it anywhere in the patch.

### Q2.2 — What is a convolution, in plain words?
A convolution is a small window of weights (say 3×3) that slides over the image, and at each position it multiplies the pixels under it by its weights and sums them into one number. Doing this across the whole image produces a "feature map" that lights up wherever that particular pattern appears. Early filters learn things like edges and color blobs; later layers combine those into textures and object parts. The network *learns* the filter weights during training rather than us hand-designing them.

### Q2.3 — What is ResNet18 and why that one?
ResNet18 is a CNN with 18 weight layers, introduced in the 2015 "Deep Residual Learning" paper. It is a sensible baseline because it is well-understood, has pretrained ImageNet weights readily available, and is small enough (~11 million parameters) to run on a CPU. We use it as a feature extractor: the convolutional body turns a 64×64×3 image into a compact vector of features, and a small final layer maps that vector to our 10 classes. It is a deliberate "boring, reliable" first model so we have a trustworthy number to beat.

### Q2.4 — What is the "residual" (skip connection) in ResNet, and why does it matter?
A residual connection lets a layer's input "skip ahead" and be added to its output, so each block only has to learn the *change* (the residual) rather than the whole transformation. This solves a real problem: very deep plain networks were actually getting *worse* because gradients vanished and signal degraded through many layers. Skip connections give gradients a clean highway back to early layers during training, so we can train deep networks reliably. It is one of the most important ideas in modern deep learning.

**Interview tip:** If you only remember one thing about ResNet, say "skip connections let you train very deep networks by giving gradients a shortcut."

### Q2.5 — What does the final layer of your network look like?
The original ResNet18 ends with a fully-connected layer that outputs 1,000 numbers (one per ImageNet class). We replace that with a new fully-connected layer that outputs **10** numbers, one per EuroSAT class. Those 10 numbers (logits) get turned into probabilities by a softmax, and the highest one is the prediction. Only this new layer (plus optionally a little of the network) is trained in the frozen-backbone baseline.

---

## Section 3 — Transfer Learning

### Q3.1 — What is transfer learning and why use it here?
Transfer learning means starting from a model already trained on a large dataset (ImageNet, ~1.2 million natural photos) and reusing what it learned instead of starting from random weights. The early-to-middle layers of an ImageNet model already know generic visual features — edges, textures, color patterns — that are useful for satellite images too. We keep those and only retrain the final decision layer for our 10 classes. With only ~27,000 images this is far more data-efficient and gives strong accuracy quickly, even on a CPU.

### Q3.2 — What does "frozen backbone" mean in your baseline?
"Freezing" means we set the pretrained convolutional layers so their weights do not update during training — they act as a fixed feature extractor. Only the new final layer (the 10-class head) has its weights learned. This makes training much faster and cheaper (we compute gradients for only a tiny fraction of the parameters), which is exactly what we need on a CPU, and it strongly resists overfitting because there is very little to overfit. The cost is a ceiling on accuracy: the features are not specialized to satellite imagery.

### Q3.3 — What is fine-tuning, and how is it different from a frozen backbone?
Fine-tuning means *unfreezing* some or all of the pretrained layers and letting them update too, usually with a small learning rate, so the features adapt to satellite imagery. A frozen backbone keeps those layers fixed and only trains the new head. Fine-tuning can squeeze out higher accuracy because the features specialize to our domain, but it is slower, needs more data to avoid overfitting, and risks "forgetting" the useful pretrained knowledge if done carelessly. Our baseline freezes first to get a clean, fast benchmark; a fine-tuned version is a natural next experiment.

**Interview tip:** Frame it as a spectrum — frozen head only → unfreeze top blocks → unfreeze everything — and say you start frozen and unfreeze gradually if accuracy plateaus.

### Q3.4 — Satellite images aren't ImageNet photos of cats and dogs. Why does transfer learning still work?
Because the *low-level* features the pretrained network learned — edges, corners, color gradients, repeating textures — are universal to almost all images, regardless of subject. A forest's canopy texture or a crop field's regular grid is built from the same visual primitives as a photo of grass or a tiled floor. Transfer learning reuses those generic building blocks; only the final "what does this combination mean" decision is retrained for land cover. It works less well for the most domain-specific cues (like multispectral bands ImageNet never saw), which is one reason fine-tuning can help.

---

## Section 4 — Training Mechanics

### Q4.1 — What loss function do you use and why?
We use **cross-entropy loss**, the standard choice for multi-class single-label classification. It measures how far the model's predicted probability for the *correct* class is from 1.0 — confidently right gives near-zero loss, confidently wrong gives a large loss. In PyTorch this is `CrossEntropyLoss`, which combines softmax and the loss in one numerically stable step, so the model outputs raw logits and the loss handles the rest. Minimizing cross-entropy pushes the model to put high probability on the true land-cover class.

### Q4.2 — What optimizer and learning rate, and what do they do?
The optimizer is the rule that updates the weights to reduce the loss; common choices are **Adam** (adaptive, forgiving) or SGD with momentum. The **learning rate** is the step size — how big an adjustment we make each update. Too high and training overshoots and diverges; too low and it crawls or gets stuck. Because we only train the small final head on a frozen backbone, a modest learning rate (e.g. around 1e-3 with Adam) works well and converges in few epochs.

**Interview tip:** "The learning rate is the single most important hyperparameter" is a true and safe thing to say.

### Q4.3 — What is batch size and what does it trade off?
Batch size is how many images the model looks at before making one weight update. Larger batches give smoother, more stable gradient estimates and use hardware more efficiently, but need more memory and can generalize slightly worse. Smaller batches are noisier but that noise can actually help find better solutions, and they fit in limited memory. On our CPU we keep the batch modest (e.g. 32 or 64) to balance speed and memory.

### Q4.4 — What is an epoch and how many do you need?
One epoch is one full pass through all the training images. Because our backbone is frozen and only a small head is learning, the model converges fast — often a handful of epochs (say 10–20) is enough, and we stop when validation accuracy stops improving. Training too few epochs underfits (the model hasn't learned enough); too many risks overfitting the head to the training set. Watching the validation curve tells us when to stop.

### Q4.5 — What is overfitting and how do you detect and prevent it here?
Overfitting is when the model memorizes the training data — including its noise — and performs well on training but poorly on new data. You detect it by watching the gap between training accuracy and validation accuracy: if training keeps climbing while validation stalls or drops, you are overfitting. In this project the frozen backbone already limits overfitting because so few parameters are trainable; we also use a held-out validation set, can add data augmentation (flips, rotations — safe for satellite imagery since orientation is arbitrary), and use early stopping. Regularization like weight decay and dropout are further options.

### Q4.6 (gotcha) — Why not just train from scratch instead of using pretrained weights?
Training from scratch means starting with random weights and learning everything from our ~27,000 images alone. That is usually worse here because deep CNNs are data-hungry — ImageNet has ~1.2 million images, roughly 45× more — so from-scratch models tend to overfit or underperform on a dataset this size, and they take far longer to train (a real problem on CPU). Pretrained weights give us a huge head start of generic visual knowledge for free. From-scratch only makes sense when you have very large domain-specific data or when your images are so unlike natural photos that pretrained features don't transfer.

---

## Section 5 — Evaluation Metrics

### Q5.1 — What is overall accuracy and what's its main weakness?
Accuracy is simply the fraction of test patches the model labels correctly. It is intuitive and easy to report, but it can be **misleading when classes are imbalanced**: if one class were 50% of the data, a lazy model that always guesses that class would score 50% while being useless on the rest. It also treats every error the same and hides *which* classes are confused. That is why we report kappa and a confusion matrix alongside it.

### Q5.2 — What is a confusion matrix and how do you read it?
A confusion matrix is a 10×10 table (one row/column per class) where each cell counts how many patches of the *true* class (row) were predicted as a given class (column). The diagonal is correct predictions; everything off the diagonal is a mistake, and it shows you *exactly which* classes get mixed up. For example, we'd expect some confusion between visually similar classes like AnnualCrop and PermanentCrop, or different built-up types. It turns a single accuracy number into a diagnostic map of the model's specific weaknesses.

**Interview tip:** Say you "read the off-diagonal" to find systematic confusions, then use that to decide what to fix next.

### Q5.3 — What is Cohen's kappa and why report it?
Cohen's kappa measures agreement between the model's predictions and the true labels **after correcting for the agreement you'd expect purely by chance**. It ranges from below 0 (worse than random) through 0 (no better than chance) up to 1 (perfect). It matters because raw accuracy is inflated by lucky/easy guesses, especially with class imbalance — kappa strips that out and gives a more honest difficulty-adjusted score. Reporting kappa alongside accuracy signals that you understand metrics, not just numbers.

### Q5.4 (gotcha) — What exactly does kappa correct for that accuracy doesn't?
Kappa corrects for **chance agreement** — the baseline level of "correct" you'd get just by guessing according to how frequent each class is. Accuracy counts a correct guess the same whether it was skill or luck; kappa effectively subtracts the luck. Concretely, kappa = (observed accuracy − expected-by-chance accuracy) / (1 − expected-by-chance accuracy). So if the classes were very imbalanced, a high raw accuracy might correspond to a much lower (more honest) kappa.

### Q5.5 (gotcha) — Your accuracy is 95%. Why might that be misleading?
A few reasons. First, if classes are imbalanced, 95% could be propped up by the easy majority classes while a rare but important class (say River or Industrial) is mostly wrong — the confusion matrix and per-class metrics would expose that. Second, and central to this project, a *random* train/test split can leak information between spatially adjacent patches, so 95% may partly reflect the model recognizing near-duplicate neighbors rather than truly generalizing — under a proper spatial split the honest number is typically lower. Third, accuracy says nothing about *which* errors happen, and some confusions matter more than others. That's why I always pair accuracy with kappa, a confusion matrix, and a spatially-aware evaluation.

### Q5.6 — What is IoU and where does it come in?
IoU (Intersection over Union) measures overlap between a predicted region and the true region: the area they share divided by the area they jointly cover, from 0 (no overlap) to 1 (perfect). It is the standard metric for *segmentation* (labeling every pixel) rather than whole-image classification. It enters in the later CNN-vs-transformer phase, where per-class IoU shows how well each land-cover type is delineated, which is more informative than a single accuracy when you care about spatial extent. For our current whole-patch classification baseline, accuracy/kappa/confusion-matrix are the right tools; IoU is the segmentation upgrade.

---

## Section 6 — Spatial Cross-Validation & Data Leakage (the signature finding)

### Q6.1 — What is data leakage, in plain words?
Data leakage is when information from the test set sneaks into training, so the model looks better than it truly is and then disappoints in the real world. It is "cheating by accident." The classic example is having near-duplicate or strongly related samples on both sides of the split. In this project the leakage is *spatial*: patches that sit next to each other on the map are very similar, so if one goes to training and its neighbor goes to test, the model has effectively already seen the answer.

### Q6.2 — What is spatial autocorrelation and why does it break a random split?
Spatial autocorrelation is the simple truth that things near each other in space tend to be similar — a forest patch is very likely surrounded by more forest. So neighboring 64×64 patches are not independent samples; they share content. A random split scatters these neighbors across train and test, so the test set is full of patches that closely resemble training patches. The model gets a high score by recognizing near-twins, which overstates how well it would do on a genuinely new, far-away region.

**Interview tip:** The phrase to drop is "the IID assumption is violated — spatial samples are not independent."

### Q6.3 — What is spatial cross-validation and how does it fix this?
Spatial cross-validation splits the data by **location** instead of randomly: you group patches into spatial blocks or regions and make sure all patches from a block go entirely to either train or test, never both. This forces the model to be evaluated on geographically separate areas it truly hasn't seen, removing the near-duplicate leakage. The headline finding of this project is that accuracy under spatial CV is **lower than under a random split** — and that lower number is the honest one. Reporting both, and explaining the gap, is the centerpiece of the project's story.

### Q6.4 (gotcha) — If spatial CV gives a lower score, isn't the random-split model "better"? Why deliberately make your numbers look worse?
The random-split model is not better — it's *fooling you*. Its higher number reflects leakage, not real-world skill; deployed on a new region it would underperform that inflated estimate. The spatial-CV number is a truthful prediction of how the model behaves where it actually matters: somewhere it hasn't memorized. Choosing the honest, lower number shows scientific maturity — you are optimizing for real generalization, not a vanity metric. This is exactly the kind of judgment that separates a portfolio project from a Kaggle leaderboard chase.

### Q6.5 — How big a gap would you expect, and what does the size tell you?
The exact gap depends on the data, but a meaningful drop (often several percentage points) is common when spatial autocorrelation is strong. A large gap tells you the random-split result was heavily leakage-inflated and the model leans on local similarity; a small gap suggests the model genuinely learned generalizable land-cover features. Either way, quantifying the gap *is* the insight — it measures how much of your apparent performance was real versus borrowed from neighbors.

---

## Section 7 — "Tell Me About Your Project" (Behavioral)

### Q7.1 — Walk me through your project end to end.
"I built a land-cover classifier on EuroSAT, a dataset of about 27,000 Sentinel-2 satellite patches, each 64×64 pixels across 10 classes like Forest, River, and Industrial. My baseline is transfer learning: I take an ImageNet-pretrained ResNet18, freeze its backbone so it acts as a fixed feature extractor, and train just a new 10-class head — which is fast enough to run on my M1 CPU. I evaluate with overall accuracy, Cohen's kappa to correct for chance, and a confusion matrix to see which classes get confused. The most interesting part is comparing a random train/test split against a spatial cross-validation split, which reveals that spatial autocorrelation inflates the random-split accuracy — so the spatially-validated number is the honest one."

**Interview tip:** Memorize this 60-second version. It hits problem, data, model, evaluation, and your unique finding in order.

### Q7.2 — What was the hardest part or biggest thing you learned?
"The biggest realization was that a perfectly normal random train/test split can quietly lie to you. My accuracy looked great, but because neighboring satellite patches are so similar, the test set was full of near-duplicates of training data. Implementing spatial cross-validation and watching the accuracy drop taught me that *how* you split data can matter as much as the model itself, and that the honest, lower number is the one worth reporting. That changed how I think about evaluation in general."

### Q7.3 — Why did you choose such a simple model? Why not a fancy transformer first?
"I chose ResNet18 with a frozen backbone on purpose, as a strong, reliable baseline. You can't tell whether a fancy model is actually helping unless you have a trustworthy number to compare it against, and a simple model gives you that quickly and cheaply — which matters on a CPU. It also surfaced the real problem worth solving, which turned out to be evaluation leakage, not model capacity. The CNN-vs-transformer benchmark with per-class IoU is the planned next phase, now that I have a clean baseline and an honest evaluation protocol to judge it against."

**Interview tip:** "Establish a trustworthy baseline before adding complexity" is a principle interviewers love to hear.

### Q7.4 — If you had more time/compute, what would you do next?
"Three things, in order. First, fine-tune the backbone instead of freezing it, since the features could specialize to satellite imagery and likely raise accuracy. Second, run the CNN-vs-transformer benchmark with per-class IoU to see if attention-based models read land texture better, especially on the confused classes. Third, push the spatial validation further with proper blocked or buffered splits and report the random-vs-spatial gap as the model's true generalization story. Everything is gated on keeping the evaluation honest."

---

## Quick Glossary
- **Logits:** the raw, un-normalized scores the final layer outputs before softmax.
- **Softmax:** turns logits into probabilities that sum to 1.
- **Backbone:** the pretrained convolutional body that extracts features.
- **Head:** the small final layer we train for our 10 classes.
- **IID:** "independent and identically distributed" — the assumption spatial data breaks.
- **Stratified split:** a split that preserves each class's proportion in every part.
