"""tutor_kb.py — Fallback knowledge base for the ML-tutor chatbot.

Embedded in the EuroSAT land-cover visual guide.
Project facts referenced throughout:
  - ResNet18, frozen-backbone transfer learning
  - 85.1% validation accuracy, Cohen's kappa 0.834
  - 27,000 images, 64x64 pixels, 10 land-cover classes
  - Top confusions: River -> Highway (81), PermanentCrop -> HerbaceousVegetation (49)
  - Guide sections: 1 image-as-numbers, 2 convolution, 3 transfer learning,
    4 training curves, 5 confusion matrix/kappa, 6 spatial-split leakage

No third-party imports. Pure standard-library Python.
"""
import re


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

KB = [
    # ----------------------------------------------------------------- 1
    {
        "keywords": [
            "overfitting", "overfit", "overfitted", "memorize", "memorizing",
            "generalize", "generalization", "train vs val", "validation gap",
            "diverge", "diverging", "too good on training", "memorizes",
            "memorize the training", "memorizing the training",
        ],
        "tab_title": "Overfitting",
        "answer": (
            "Overfitting is when a model memorizes the training images instead of "
            "learning patterns that generalize to new images. You spot it when the "
            "training accuracy keeps climbing but the validation accuracy stalls or "
            "drops, so a gap opens up between the two curves. In this EuroSAT "
            "project we froze the ResNet18 backbone, which is one of the strongest "
            "defenses against overfitting because far fewer weights can be tuned to "
            "memorize. The result was a healthy 85.1% validation accuracy that "
            "tracked the training curve closely instead of diverging. If we had "
            "unfrozen everything on only 27,000 small 64x64 images, the model could "
            "easily have started memorizing and the validation curve would have "
            "fallen behind. The cure is more data, regularization, early stopping, "
            "or freezing layers — exactly the strategy used here."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Train accuracy keeps rising; validation plateaus — the gap is overfitting</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
function loss(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  // axes
  x.strokeStyle="#2a3340";x.lineWidth=1;
  x.beginPath();x.moveTo(50,20);x.lineTo(50,250);x.lineTo(490,250);x.stroke();
  x.fillStyle="#9aa5b1";x.font="11px Arial";
  x.fillText("accuracy",6,20);x.fillText("epochs",440,268);
  var prog=(t%240)/240; // 0..1 sweep
  var n=Math.floor(prog*100)+1;
  // train curve: rises toward ~0.99
  x.strokeStyle="#ffa726";x.lineWidth=2.5;x.beginPath();
  for(var i=0;i<=n;i++){
    var p=i/100;
    var tr=0.99*(1-Math.exp(-3.2*p));
    var px=50+p*440, py=250-tr*220;
    if(i==0)x.moveTo(px,py);else x.lineTo(px,py);
  }
  x.stroke();
  // val curve: rises then plateaus ~0.851
  x.strokeStyle="#4fc3f7";x.lineWidth=2.5;x.beginPath();
  for(var j=0;j<=n;j++){
    var q=j/100;
    var va=0.851*(1-Math.exp(-3.0*q)) - 0.05*Math.max(0,q-0.6);
    var vx=50+q*440, vy=250-va*220;
    if(j==0)x.moveTo(vx,vy);else x.lineTo(vx,vy);
  }
  x.stroke();
  // gap shading at current point
  var cp=n/100;
  var trc=0.99*(1-Math.exp(-3.2*cp));
  var vac=0.851*(1-Math.exp(-3.0*cp)) - 0.05*Math.max(0,cp-0.6);
  var gx=50+cp*440;
  x.strokeStyle="rgba(255,255,255,0.25)";x.setLineDash([4,4]);
  x.beginPath();x.moveTo(gx,250-trc*220);x.lineTo(gx,250-vac*220);x.stroke();
  x.setLineDash([]);
  // legend
  x.fillStyle="#ffa726";x.fillRect(330,30,12,12);x.fillStyle="#e8edf2";x.fillText("train",348,40);
  x.fillStyle="#4fc3f7";x.fillRect(400,30,12,12);x.fillStyle="#e8edf2";x.fillText("val (85.1%)",418,40);
  if(cp>0.62){
    var midy=(250-trc*220+250-vac*220)/2;
    x.fillStyle="#ff6b6b";x.font="bold 12px Arial";x.fillText("GAP",gx+6,midy);
  }
  t++;requestAnimationFrame(loss);
}
loss();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 2
    {
        "keywords": [
            "learning rate", "step size", "lr", "too big", "too small",
            "overshoot", "step", "optimizer step", "stride length",
        ],
        "tab_title": "Learning Rate",
        "answer": (
            "The learning rate controls how big a step the optimizer takes each "
            "time it updates the weights, like the stride length of a ball rolling "
            "down into a valley of low loss. If the rate is too small, training "
            "crawls and takes forever to reach the bottom; if it is too big, the "
            "ball overshoots, bounces off the walls, and may never settle. A good "
            "learning rate descends quickly and then eases into the minimum. In "
            "this project we trained only the new classifier head on top of a "
            "frozen ResNet18, so a moderate learning rate let the model reach 85.1% "
            "validation accuracy smoothly. Tuning this single number is often the "
            "highest-leverage knob you have. Watch the animation: the small step "
            "creeps, the big step overshoots, and the good step lands cleanly."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Same loss valley, three learning rates: too small crawls, too big overshoots</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
function fy(px){var u=(px-260)/180; return 250-( (1-Math.exp(-u*u*0.0009*0))+0 );}
// loss curve: parabola
function curveY(px){var u=(px-260)/170; return 70+ u*u*150;}
var balls=[
 {x0:90,lr:0.02,col:"#66bb6a",lab:"small lr"},
 {x0:430,lr:0.30,col:"#ffa726",lab:"big lr"},
 {x0:120,lr:0.10,col:"#4fc3f7",lab:"good lr"}
];
var state=balls.map(function(b){return {x:b.x0,v:0};});
function reset(){state=balls.map(function(b){return {x:b.x0,v:0};});}
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  // draw valley
  x.strokeStyle="#2a3340";x.lineWidth=2;x.beginPath();
  for(var px=40;px<=480;px+=2){var py=curveY(px);if(px==40)x.moveTo(px,py);else x.lineTo(px,py);}
  x.stroke();
  // minimum marker
  x.strokeStyle="rgba(255,255,255,0.15)";x.setLineDash([3,3]);
  x.beginPath();x.moveTo(260,70);x.lineTo(260,260);x.stroke();x.setLineDash([]);
  // physics: gradient of parabola ~ slope
  for(var i=0;i<balls.length;i++){
    var b=balls[i],s=state[i];
    var slope=2*(s.x-260)/170/170*150; // d(curveY)/dx
    // gradient descent on x toward 260
    var grad=(s.x-260);
    s.x -= b.lr*grad*0.08;
    // clamp
    if(s.x<45)s.x=45; if(s.x>475)s.x=475;
    var by=curveY(s.x)-8;
    x.beginPath();x.arc(s.x,by,7,0,7);x.fillStyle=b.col;x.fill();
    x.fillStyle=b.col;x.font="11px Arial";x.fillText(b.lab,s.x-22,by-12);
  }
  t++;
  if(t%180===0)reset();
  requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 3
    {
        "keywords": [
            "softmax", "logits", "logit", "probabilities", "probability",
            "normalize scores", "class scores", "confidence", "turn into probabilities",
            "confident", "how sure",
        ],
        "tab_title": "Softmax",
        "answer": (
            "Softmax is the final step that turns the network's raw scores, called "
            "logits, into probabilities that add up to 100%. The network outputs one "
            "logit per class — here 10 logits for the 10 EuroSAT land-cover types — "
            "and these can be any positive or negative numbers. Softmax exponentiates "
            "each logit and divides by the total, so larger logits become larger "
            "probabilities and the whole set sums to one. That is how the model can "
            "say something like 'River sixty-two percent, Highway thirty percent' "
            "instead of meaningless raw numbers, and the biggest probability becomes "
            "the predicted class. When River "
            "and Highway logits are close, softmax produces a near-tie, which is "
            "exactly why this project saw 81 River images misclassified as Highway. "
            "Watch the bars normalize from raw logits into a clean probability "
            "distribution."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Raw logits (left) become probabilities summing to 1 (right) via softmax</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var names=["River","Highway","Forest","Crop","Sea"];
var logits=[3.1,2.4,0.5,-0.3,-1.2];
var ex=logits.map(function(l){return Math.exp(l);});
var sum=ex.reduce(function(a,b){return a+b;},0);
var probs=ex.map(function(e){return e/sum;});
// for raw display normalize logits to 0..1 for bar height visualization
var minL=Math.min.apply(null,logits),maxL=Math.max.apply(null,logits);
var rawH=logits.map(function(l){return (l-minL)/(maxL-minL);});
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var phase=(t%300)/300; // 0..1 morph
  var m=phase<0.15?0:(phase>0.85?1:(phase-0.15)/0.7);
  m=m*m*(3-2*m); // smoothstep
  var n=names.length, bw=70, gap=20, x0=40;
  x.fillStyle="#9aa5b1";x.font="12px Arial";
  x.fillText(m<0.5?"raw logits":"probabilities (sum=1.00)",170,26);
  for(var i=0;i<n;i++){
    var h = rawH[i]*(1-m) + probs[i]*2.0*m; // probs scaled so visible
    if(h>1)h=1;
    var bh=h*180;
    var bx=x0+i*(bw+gap), by=250-bh;
    var col=i==0?"#4fc3f7":(i==1?"#ffa726":"#3a4656");
    x.fillStyle=col;x.fillRect(bx,by,bw,bh);
    x.fillStyle="#e8edf2";x.font="11px Arial";
    x.fillText(names[i],bx+4,268);
    var val = m<0.5 ? logits[i].toFixed(1) : (probs[i]*100).toFixed(0)+"%";
    x.fillStyle="#e8edf2";x.font="bold 11px Arial";
    x.fillText(val,bx+18,by-6);
  }
  // formula
  x.fillStyle="#66bb6a";x.font="13px Courier";
  x.fillText("softmax(z)_i = e^z_i / sum(e^z)",120,292);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 4
    {
        "keywords": [
            "epoch", "batch", "batches", "mini-batch", "minibatch",
            "epoch vs batch", "batch size", "iteration", "pass through data",
            "pass over data", "full pass", "one pass",
        ],
        "tab_title": "Epoch vs Batch",
        "answer": (
            "A batch is a small group of images the model looks at together before "
            "updating its weights once, and an epoch is one full pass through the "
            "entire training set. Because GPUs cannot hold all data at once, we feed "
            "images in batches — say 32 or 64 at a time — and after each batch the "
            "optimizer takes a step. When every image has been seen exactly once, "
            "that completes one epoch, and training usually runs for many epochs. In "
            "this EuroSAT project the training split is a portion of the 27,000 "
            "64x64 images, so each epoch consists of hundreds of batches. More "
            "batches per epoch means more weight updates and smoother learning. The "
            "animation shows a grid of images being consumed batch by batch until the "
            "whole grid — one epoch — is done, then it repeats."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Images consumed batch-by-batch; one full grid = one epoch</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var cols=10,rows=6,cell=30,pad=6,ox=60,oy=40;
var total=cols*rows;
var batch=5; // images per batch
var consumed=0,epoch=1,frameCount=0;
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  x.fillStyle="#9aa5b1";x.font="13px Arial";
  x.fillText("Epoch "+epoch,ox,26);
  x.fillText("batch size = "+batch,ox+160,26);
  x.fillText("processed "+consumed+" / "+total,ox+300,26);
  for(var i=0;i<total;i++){
    var r=Math.floor(i/cols),c=i%cols;
    var cx=ox+c*(cell+pad), cy=oy+r*(cell+pad);
    var done=i<consumed;
    var inCurrent = i>=consumed && i<consumed+batch;
    x.fillStyle=done?"#66bb6a":(inCurrent?"#ffa726":"#2a3340");
    x.fillRect(cx,cy,cell,cell);
    // tiny "image" texture
    x.fillStyle="rgba(15,20,25,0.35)";
    x.fillRect(cx+4,cy+4,cell-8,(cell-8)/2);
  }
  // current batch label
  x.fillStyle="#ffa726";x.font="12px Arial";
  if(consumed<total)x.fillText("-> current batch (one weight update)",ox,oy+rows*(cell+pad)+24);
  else x.fillText("epoch complete -> next epoch",ox,oy+rows*(cell+pad)+24);
  // legend
  x.fillStyle="#66bb6a";x.fillRect(ox,oy+rows*(cell+pad)+36,12,12);x.fillStyle="#e8edf2";x.font="11px Arial";x.fillText("seen",ox+16,oy+rows*(cell+pad)+46);
  x.fillStyle="#ffa726";x.fillRect(ox+90,oy+rows*(cell+pad)+36,12,12);x.fillStyle="#e8edf2";x.fillText("processing",ox+106,oy+rows*(cell+pad)+46);
  x.fillStyle="#2a3340";x.fillRect(ox+200,oy+rows*(cell+pad)+36,12,12);x.fillStyle="#e8edf2";x.fillText("waiting",ox+216,oy+rows*(cell+pad)+46);
  frameCount++;
  if(frameCount%18===0){
    consumed+=batch;
    if(consumed>=total){consumed=0;epoch++;if(epoch>3)epoch=1;}
  }
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 5
    {
        "keywords": [
            "gradient descent", "gradient", "descent", "downhill", "minimize loss",
            "slope", "backprop direction", "follow the gradient",
            "backprop", "backpropagation", "back propagation",
            "how does it learn", "how does the model learn", "weights update",
            "update the weights", "weight update", "update weights",
        ],
        "tab_title": "Gradient Descent",
        "answer": (
            "Gradient descent is the core algorithm that trains the network by "
            "repeatedly walking downhill on the loss surface. At each point the "
            "gradient tells you the direction of steepest increase, so we step in "
            "the opposite direction to make the loss smaller. Take a step, measure "
            "the new slope, step again — over thousands of steps the weights settle "
            "near a minimum where predictions are good. In this project gradient "
            "descent only had to tune the new classifier head sitting on the frozen "
            "ResNet18, which made the descent fast and stable, landing at 85.1% "
            "validation accuracy. The size of each step is the learning rate, and the "
            "direction always comes from the gradient. The animation shows a point "
            "sliding down a curve with arrows marking each downhill step until it "
            "reaches the bottom."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Each arrow steps opposite the gradient — downhill toward minimum loss</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
function curveY(px){var u=(px-300)/150; return 60+ u*u*160;}
function deriv(px){return 2*(px-300)/150/150*160;}
var pos=70, lr=14, settled=false, hist=[];
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  // curve
  x.strokeStyle="#2a3340";x.lineWidth=2;x.beginPath();
  for(var px=40;px<=500;px+=2){var py=curveY(px);if(px==40)x.moveTo(px,py);else x.lineTo(px,py);}
  x.stroke();
  x.fillStyle="#9aa5b1";x.font="12px Arial";x.fillText("loss",46,60);
  // trail of past steps
  for(var i=0;i<hist.length;i++){
    x.fillStyle="rgba(79,195,247,0.3)";
    x.beginPath();x.arc(hist[i],curveY(hist[i]),3,0,7);x.fill();
  }
  // current ball
  var by=curveY(pos);
  x.beginPath();x.arc(pos,by,8,0,7);x.fillStyle="#4fc3f7";x.fill();
  // gradient arrow (downhill direction)
  var g=deriv(pos);
  var dir = g>0?-1:1;
  var ax=pos, ay=by-22;
  x.strokeStyle="#ffa726";x.lineWidth=2.5;
  x.beginPath();x.moveTo(ax,ay);x.lineTo(ax+dir*30,ay);x.stroke();
  // arrowhead
  x.beginPath();x.moveTo(ax+dir*30,ay);x.lineTo(ax+dir*22,ay-5);x.lineTo(ax+dir*22,ay+5);x.closePath();x.fillStyle="#ffa726";x.fill();
  x.fillStyle="#ffa726";x.font="11px Arial";x.fillText("-gradient",pos-20,ay-10);
  // step every N frames
  if(t%24===0 && !settled){
    hist.push(pos);
    pos -= lr*Math.sign(g)*Math.min(1,Math.abs(g)*8+0.2);
    if(Math.abs(pos-300)<4){settled=true;}
  }
  if(settled){
    x.fillStyle="#66bb6a";x.font="bold 13px Arial";x.fillText("minimum reached",pos-50,by-30);
    if(t%140===0){pos=70;settled=false;hist=[];}
  }
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 6
    {
        "keywords": [
            "normalize", "normalization", "normalisation", "mean zero", "mean 0",
            "standardize", "standardization", "scale pixels", "preprocess",
            "subtract mean", "why normalize", "preprocessing", "scaling",
        ],
        "tab_title": "Why Normalize",
        "answer": (
            "Normalizing means shifting and scaling the pixel values so they are "
            "centered around zero with a consistent spread, instead of raw 0-255 "
            "brightness numbers. Raw pixels are large and lopsided, which makes the "
            "loss surface stretched and harder for gradient descent to traverse "
            "smoothly. By subtracting the dataset mean and dividing by the standard "
            "deviation, every input channel sits in a comparable, well-behaved range, "
            "so training is faster and more stable. This is especially important here "
            "because we reused ImageNet-pretrained ResNet18 weights, which expect "
            "inputs normalized the same way the original network was trained. Feeding "
            "the 64x64 EuroSAT images with matching normalization let the frozen "
            "features transfer cleanly and reach 85.1% accuracy. The animation shows "
            "a pixel-value distribution sliding from a high, skewed position to a "
            "tidy bell centered on zero."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Raw pixel distribution shifts and rescales to mean 0, unit spread</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  // axis
  x.strokeStyle="#2a3340";x.lineWidth=1;
  x.beginPath();x.moveTo(30,250);x.lineTo(500,250);x.stroke();
  // zero line target
  var zeroX=265;
  x.strokeStyle="rgba(102,187,106,0.4)";x.setLineDash([4,4]);
  x.beginPath();x.moveTo(zeroX,40);x.lineTo(zeroX,250);x.stroke();x.setLineDash([]);
  x.fillStyle="#66bb6a";x.font="11px Arial";x.fillText("0",zeroX-3,266);
  var phase=(t%300)/300;
  var m=phase<0.15?0:(phase>0.85?1:(phase-0.15)/0.7);
  m=m*m*(3-2*m);
  // raw: centered high (mean ~ +120 in pixel space -> rightish), wide
  // normalized: centered at zero, narrower
  var rawMean=410, rawSd=70;
  var nMean=zeroX, nSd=42;
  var mean=rawMean+(nMean-rawMean)*m;
  var sd=rawSd+(nSd-rawSd)*m;
  // draw bell
  x.strokeStyle=m<0.5?"#ffa726":"#4fc3f7";x.lineWidth=2.5;x.beginPath();
  for(var px=30;px<=500;px+=2){
    var u=(px-mean)/sd;
    var y=250-180*Math.exp(-0.5*u*u);
    if(px==30)x.moveTo(px,y);else x.lineTo(px,y);
  }
  x.stroke();
  // fill under
  x.lineTo(500,250);x.lineTo(30,250);x.closePath();
  x.fillStyle=m<0.5?"rgba(255,167,38,0.15)":"rgba(79,195,247,0.15)";x.fill();
  // mean marker
  x.strokeStyle="#e8edf2";x.lineWidth=1.5;
  x.beginPath();x.moveTo(mean,70);x.lineTo(mean,250);x.stroke();
  x.fillStyle="#e8edf2";x.font="11px Arial";
  x.fillText("mean",mean-14,64);
  // labels
  x.fillStyle="#9aa5b1";x.font="12px Arial";
  x.fillText(m<0.5?"raw pixels 0..255 (skewed, large)":"normalized: mean 0, unit std",150,28);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 7
    {
        "keywords": [
            "kappa", "cohen", "cohen's kappa", "chance agreement", "chance",
            "kappa vs accuracy", "0.834", "agreement", "balanced metric",
            "agree", "by chance", "random guessing",
        ],
        "tab_title": "Kappa vs Accuracy",
        "answer": (
            "Cohen's kappa measures how much better the model agrees with the truth "
            "than it would by random guessing alone. Plain accuracy can look "
            "flattering when some classes are common, because you would get many "
            "right just by guessing the majority class. Kappa subtracts that expected "
            "chance agreement and then rescales, so a kappa of 1.0 is perfect and 0 "
            "means no better than chance. In this project the accuracy was 85.1% but "
            "the kappa was 0.834, slightly lower because it discounts the easy "
            "agreement you would get for free across the 10 land-cover classes. That "
            "0.834 is still strong and confirms the model is genuinely skillful, not "
            "just lucky on frequent classes. The animation shows the chance-agreement "
            "baseline filling up first, then kappa rescaling the remaining real skill "
            "above that baseline."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Kappa removes the chance-agreement baseline, then rescales real skill</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var acc=0.851, pe=0.10; // expected chance agreement (~1/10 classes)
var kappa=(acc-pe)/(1-pe); // ~0.834
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var phase=(t%360)/360;
  // bar 1: accuracy
  var bx=90,bw=110, base=250, full=200;
  x.fillStyle="#9aa5b1";x.font="12px Arial";
  x.fillText("Accuracy",bx+10,base+22);
  x.fillText("Cohen's kappa",bx+200,base+22);
  // accuracy bar grows 0..0.4 of phase
  var a1=Math.min(1,phase/0.35);
  var ah=acc*full*a1;
  x.fillStyle="#4fc3f7";x.fillRect(bx,base-ah,bw,ah);
  x.fillStyle="#e8edf2";x.font="bold 13px Arial";
  if(a1>=1)x.fillText((acc*100).toFixed(1)+"%",bx+30,base-ah-8);
  // chance baseline overlay appears phase 0.35..0.55
  if(phase>0.35){
    var pe_h=pe*full*Math.min(1,(phase-0.35)/0.2);
    x.fillStyle="rgba(255,167,38,0.85)";x.fillRect(bx,base-pe_h,bw,pe_h);
    x.fillStyle="#ffa726";x.font="11px Arial";
    if(phase>0.5)x.fillText("chance",bx+4,base-2);
  }
  // arrow showing subtraction
  if(phase>0.55){
    x.fillStyle="#9aa5b1";x.font="20px Arial";x.fillText("->",bx+150,base-90);
    x.fillStyle="#9aa5b1";x.font="11px Arial";x.fillText("remove",bx+150,base-72);
    x.fillText("chance,",bx+150,base-60);x.fillText("rescale",bx+150,base-48);
  }
  // kappa bar grows phase 0.6..1
  var bx2=bx+200;
  if(phase>0.6){
    var k1=Math.min(1,(phase-0.6)/0.35);
    var kh=kappa*full*k1;
    x.fillStyle="#66bb6a";x.fillRect(bx2,base-kh,bw,kh);
    x.fillStyle="#e8edf2";x.font="bold 13px Arial";
    if(k1>=1)x.fillText(kappa.toFixed(3),bx2+28,base-kh-8);
  }
  // formula
  x.fillStyle="#66bb6a";x.font="13px Courier";
  x.fillText("k = (acc - p_chance) / (1 - p_chance)",110,40);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 8
    {
        "keywords": [
            "fine-tuning", "fine tuning", "finetune", "fine-tune", "frozen backbone",
            "frozen", "freeze", "unfreeze", "unlock layers", "transfer learning",
            "backbone",
        ],
        "tab_title": "Frozen vs Fine-tune",
        "answer": (
            "With a frozen backbone you keep all the pretrained ResNet18 layers "
            "locked and only train a fresh classifier head on top, which is fast, "
            "needs little data, and resists overfitting. Fine-tuning instead unlocks "
            "some or all of those backbone layers so their weights can also adapt to "
            "your specific images, which can squeeze out more accuracy but needs more "
            "data and care. This project used the frozen-backbone approach: the "
            "ImageNet features stayed fixed and only the head learned the 10 EuroSAT "
            "classes, reaching 85.1% accuracy and kappa 0.834 on just 27,000 small "
            "images. Freezing was the right call because unfreezing everything on "
            "this modest dataset would risk overfitting. A common middle path is to "
            "freeze early layers and fine-tune only the last few. The animation shows "
            "layers staying locked while the head trains, then optionally unlocking "
            "one block at a time."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Frozen backbone trains only the head; fine-tuning unlocks blocks over time</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var layers=["Input","Block1","Block2","Block3","Block4","Head"];
var nL=layers.length;
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var phase=(t%480)/480;
  // how many backbone blocks unlocked: 0..4 over phase 0.5..1
  var unlocked=0;
  if(phase>0.5)unlocked=Math.floor((phase-0.5)/0.1);
  if(unlocked>4)unlocked=4;
  var bw=70,gap=12,x0=30,by=110,bh=90;
  for(var i=0;i<nL;i++){
    var bx=x0+i*(bw+gap);
    var isHead=(i===nL-1);
    // backbone blocks index 1..4 correspond to i 1..4
    var blockIdx=i; // for blocks 1..4 -> i 1..4
    var frozen = !isHead && i>0 && (5-i) > unlocked; // later blocks unlock first
    // Input always frozen-ish (data)
    if(i===0){
      x.fillStyle="#2a3340";
    } else if(isHead){
      x.fillStyle="#66bb6a"; // head always training
    } else if(frozen){
      x.fillStyle="#3a4656";
    } else {
      x.fillStyle="#ffa726"; // unlocked / training
    }
    x.fillRect(bx,by,bw,bh);
    x.fillStyle="#0f1419";x.font="bold 12px Arial";
    x.fillText(layers[i],bx+8,by+bh/2);
    // lock / training icon
    if(!isHead && i>0){
      if(frozen){
        x.fillStyle="#9aa5b1";x.font="16px Arial";x.fillText("\u{1F512}",bx+bw/2-8,by-8);
      } else {
        x.fillStyle="#ffa726";x.font="11px Arial";x.fillText("training",bx+10,by-8);
      }
    }
    if(isHead){x.fillStyle="#66bb6a";x.font="11px Arial";x.fillText("training",bx+12,by-8);}
    // connecting arrow
    if(i<nL-1){x.strokeStyle="#2a3340";x.lineWidth=2;x.beginPath();x.moveTo(bx+bw,by+bh/2);x.lineTo(bx+bw+gap,by+bh/2);x.stroke();}
  }
  // gradient flow animation along trainable parts (right to left into unlocked)
  var flowX = 480 - (t%120)/120*((unlocked+1)*(bw+gap));
  x.fillStyle="rgba(255,167,38,0.6)";
  x.beginPath();x.arc(flowX,by+bh+18,4,0,7);x.fill();
  x.fillStyle="#9aa5b1";x.font="11px Arial";x.fillText("backprop / gradients",340,by+bh+22);
  // mode label
  x.fillStyle="#e8edf2";x.font="14px Arial";
  if(phase<0.5)x.fillText("FROZEN BACKBONE: only Head trains (this project)",60,40);
  else x.fillText("FINE-TUNING: unlocking "+unlocked+" block(s) from the top",60,40);
  // legend
  x.fillStyle="#3a4656";x.fillRect(60,250,14,14);x.fillStyle="#e8edf2";x.font="11px Arial";x.fillText("frozen",78,262);
  x.fillStyle="#ffa726";x.fillRect(150,250,14,14);x.fillStyle="#e8edf2";x.fillText("unlocked",168,262);
  x.fillStyle="#66bb6a";x.fillRect(260,250,14,14);x.fillStyle="#e8edf2";x.fillText("head (always trains)",278,262);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 9
    {
        "keywords": [
            "spatial autocorrelation", "spatial leakage", "leakage", "nearby",
            "neighboring patches", "neighbouring patches", "same farm",
            "adjacent", "tobler", "leak", "inflated", "inflate",
            "why is accuracy inflated", "optimistic", "too high accuracy",
        ],
        "tab_title": "Spatial Autocorrelation",
        "answer": (
            "Spatial autocorrelation is Tobler's first law of geography: near things "
            "are more alike than distant things. Two satellite patches cut from the "
            "same farm look almost identical — same crop, same soil, same shadows. If "
            "one of those twin patches lands in the training set and the other lands "
            "in validation, the model can simply recognize the place instead of "
            "learning the underlying land-cover concept. That is spatial leakage, and "
            "it quietly inflates your scores. A random 80/20 split scatters patches "
            "everywhere, so adjacent twins routinely straddle the train and val sets, "
            "which is exactly why this project's 85.1% random-split accuracy and 0.834 "
            "kappa may be optimistic. The honest fix is a spatially-separated split "
            "that keeps a whole contiguous region out for validation, so no patch in "
            "val has a near-twin in train. The animation shows a map of patches: in "
            "random mode adjacent blue and orange twins flash 'LEAK!', then it toggles "
            "to a clean contiguous orange region with no twins — the honest split."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Random split leaks adjacent twins across train/val; spatial split keeps val honest</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var cols=12,rows=8,cell=30,pad=3,ox=70,oy=46;
var rnd=[];
var seed=12345;
function rng(){seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;}
for(var i=0;i<cols*rows;i++){rnd.push(rng()<0.2?1:0);}
// pick an adjacent twin pair for the leak highlight (random mode)
var leakA=3*cols+4, leakB=3*cols+5;
rnd[leakA]=0; rnd[leakB]=1;
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var spatial=(Math.floor(t/240)%2)===1;
  x.font="14px Arial";
  x.fillStyle=spatial?"#66bb6a":"#ffa726";
  x.fillText(spatial?"SPATIAL SPLIT (honest)":"RANDOM SPLIT (leaks)",ox,30);
  for(var r=0;r<rows;r++){
    for(var c=0;c<cols;c++){
      var idx=r*cols+c;
      var cx=ox+c*(cell+pad), cy=oy+r*(cell+pad);
      var isVal;
      if(spatial){ isVal = c>=cols-4; }
      else { isVal = rnd[idx]===1; }
      x.fillStyle=isVal?"#ffa726":"#4fc3f7";
      x.fillRect(cx,cy,cell,cell);
      x.fillStyle="rgba(15,20,25,0.30)";
      x.fillRect(cx+4,cy+4,cell-8,(cell-8)/2);
    }
  }
  if(!spatial){
    var ar=Math.floor(leakA/cols),ac=leakA%cols;
    var br=Math.floor(leakB/cols),bc=leakB%cols;
    var axp=ox+ac*(cell+pad),ayp=oy+ar*(cell+pad);
    var bxp=ox+bc*(cell+pad),byp=oy+br*(cell+pad);
    var pulse=2+2*Math.sin(t*0.18);
    x.strokeStyle="#ff6b6b";x.lineWidth=pulse;
    x.strokeRect(axp-2,ayp-2,cell+4,cell+4);
    x.strokeRect(bxp-2,byp-2,cell+4,cell+4);
    x.fillStyle="#ff6b6b";x.font="bold 13px Arial";
    x.fillText("LEAK!",bxp+cell+8,byp+cell/2+4);
    x.fillStyle="#9aa5b1";x.font="11px Arial";
    x.fillText("same farm: one in train, twin in val",ox,oy+rows*(cell+pad)+18);
  } else {
    x.fillStyle="#9aa5b1";x.font="11px Arial";
    x.fillText("val is one contiguous region: no adjacent twins in train",ox,oy+rows*(cell+pad)+18);
  }
  x.fillStyle="#4fc3f7";x.fillRect(ox,oy+rows*(cell+pad)+28,12,12);
  x.fillStyle="#e8edf2";x.font="11px Arial";x.fillText("train",ox+16,oy+rows*(cell+pad)+38);
  x.fillStyle="#ffa726";x.fillRect(ox+80,oy+rows*(cell+pad)+28,12,12);
  x.fillStyle="#e8edf2";x.fillText("val",ox+96,oy+rows*(cell+pad)+38);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 10
    {
        "keywords": [
            "cross-validation", "cross validation", "k-fold", "kfold", "k fold",
            "folds", "fold", "validation strategy", "groupkfold",
            "split strategy", "how to split",
        ],
        "tab_title": "K-Fold Cross-Validation",
        "answer": (
            "Instead of trusting a single 80/20 split, k-fold cross-validation slices "
            "the data into k equal folds and rotates through them: each fold takes a "
            "turn as validation while the other k-1 folds train, so every image is "
            "validated exactly once. You then report the mean accuracy plus or minus "
            "the standard deviation across folds, which is far more trustworthy than "
            "one lucky or unlucky split. For geospatial data the folds must be spatial "
            "blocks rather than random shuffles — that is what GroupKFold gives you, "
            "grouping nearby patches together so twins never split across a boundary "
            "and the score stays honest. This EuroSAT project uses a single spatial "
            "holdout rather than full spatial k-fold, purely for CPU-budget reasons, "
            "since each fold means retraining the model. The animation shows a bar of "
            "five segments where the orange validation block rotates through every "
            "position, fold by fold, then reports a mean accuracy at the end."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Validation fold rotates through all 5 positions; mean +/- std is the honest score</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var K=5;
var accs=[0.847,0.861,0.839,0.855,0.852];
var holdFrames=48;
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var cycle=K*holdFrames+holdFrames; // last slot = summary
  var ph=t%cycle;
  var foldIdx=Math.floor(ph/holdFrames);
  var summary=foldIdx>=K;
  var valFold=summary?-1:foldIdx;
  var bw=88,gap=8,x0=40,by=110,bh=70;
  x.fillStyle="#e8edf2";x.font="14px Arial";
  x.fillText(summary?"all folds done":("fold "+(foldIdx+1)+" / "+K),x0,60);
  for(var i=0;i<K;i++){
    var bx=x0+i*(bw+gap);
    var isVal=(i===valFold);
    x.fillStyle=isVal?"#ffa726":"#4fc3f7";
    x.fillRect(bx,by,bw,bh);
    x.fillStyle=isVal?"#0f1419":"#e8edf2";x.font="bold 12px Arial";
    x.fillText(isVal?"VAL":"train",bx+bw/2-16,by+bh/2+4);
    x.fillStyle="#9aa5b1";x.font="11px Arial";
    x.fillText("fold "+(i+1),bx+bw/2-18,by+bh+16);
    if(!summary && i===valFold){
      x.fillStyle="#ffa726";x.font="11px Arial";
      x.fillText((accs[i]*100).toFixed(1)+"%",bx+bw/2-16,by-8);
    }
  }
  if(summary){
    var mean=0;for(var j=0;j<K;j++)mean+=accs[j];mean/=K;
    var v=0;for(var k=0;k<K;k++)v+=(accs[k]-mean)*(accs[k]-mean);
    var sd=Math.sqrt(v/K);
    x.fillStyle="#66bb6a";x.font="bold 18px Arial";
    x.fillText("mean = "+(mean*100).toFixed(1)+"%  +/- "+(sd*100).toFixed(1)+"%",x0+30,by+bh+70);
    x.fillStyle="#9aa5b1";x.font="11px Arial";
    x.fillText("more trustworthy than one single split",x0+60,by+bh+90);
  } else {
    x.fillStyle="#9aa5b1";x.font="11px Arial";
    x.fillText("each fold takes one turn as validation",x0,by+bh+70);
    x.fillText("for geodata use spatial blocks (GroupKFold)",x0,by+bh+90);
  }
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 11
    {
        "keywords": [
            "utm", "coordinates", "coordinate", "geotiff", "geo-tiff",
            "georeferenced", "georeference", "crs", "projection", "easting",
            "northing", "epsg", "tiepoint", "tif", "tiff", "metadata", "location",
        ],
        "tab_title": "GeoTIFF Coordinates",
        "answer": (
            "A GeoTIFF is an ordinary image file that also embeds where each pixel "
            "sits on Earth, stored in metadata tags. Three tags do the heavy lifting: "
            "a tiepoint, which pins one corner of the image to a real-world "
            "coordinate; a pixel scale, which says how many meters each pixel covers; "
            "and a CRS or EPSG code, which names the map projection used, such as UTM "
            "zone 32N. Together they let software convert any pixel's row and column "
            "into an easting and northing on the ground. This project reads those tags "
            "from every one of the 27,000 EuroSAT patches to place each one at its true "
            "location on a map, and that map is what makes spatially-separated folds "
            "possible — you can only hold out a contiguous region if you know where "
            "each patch actually is. The animation shows a satellite patch with its "
            "tiepoint, scale, and EPSG tags floating out, then an arrow dropping the "
            "patch onto a map of Europe at its coordinates, repeating for new patches."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">GeoTIFF tags (tiepoint, scale, EPSG) place each patch at its real coordinates</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var spots=[[360,90],[400,140],[330,170],[420,95],[355,200]];
var idx=0;
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  // simple Europe-ish outline on the right
  x.strokeStyle="#2a3340";x.lineWidth=2;x.beginPath();
  x.moveTo(300,70);x.lineTo(360,55);x.lineTo(420,70);x.lineTo(455,110);
  x.lineTo(440,160);x.lineTo(400,210);x.lineTo(350,225);x.lineTo(315,190);
  x.lineTo(300,140);x.closePath();x.stroke();
  x.fillStyle="rgba(79,195,247,0.06)";x.fill();
  x.fillStyle="#9aa5b1";x.font="11px Arial";x.fillText("map (UTM)",315,250);
  // already-placed patches
  for(var i=0;i<idx;i++){
    x.fillStyle="#66bb6a";x.fillRect(spots[i][0]-4,spots[i][1]-4,8,8);
  }
  var cycle=180;
  var ph=(t%cycle)/cycle;
  var sp=spots[idx];
  // source patch on the left
  var sx=70,sy=110,ps=64;
  x.fillStyle="#3a4656";x.fillRect(sx,sy,ps,ps);
  for(var g=0;g<4;g++){
    x.fillStyle=g%2?"#4a5a6e":"#2f3b49";
    x.fillRect(sx+ (g%2)*ps/2, sy+ Math.floor(g/2)*ps/2, ps/2, ps/2);
  }
  x.fillStyle="#9aa5b1";x.font="11px Arial";x.fillText("satellite patch",sx-2,sy-8);
  // floating tags
  var tags=["tiepoint: 411000E, 5650000N","scale: 10 m/px","EPSG: 32632 (UTM 32N)"];
  for(var k=0;k<tags.length;k++){
    var appear=Math.min(1,Math.max(0,(ph-0.05-k*0.1)/0.15));
    if(appear<=0)continue;
    var ty=sy+ps+24+k*22;
    var tx=sx + appear*30;
    x.globalAlpha=appear;
    x.fillStyle="#ffa726";x.font="11px Courier";
    x.fillText(tags[k],tx,ty);
    x.globalAlpha=1;
  }
  // arrow placing patch onto map after tags shown
  if(ph>0.45){
    var fly=Math.min(1,(ph-0.45)/0.4);
    var fx=sx+ps/2 + (sp[0]-(sx+ps/2))*fly;
    var fy=sy+ps/2 + (sp[1]-(sy+ps/2))*fly;
    x.fillStyle="#4fc3f7";x.fillRect(fx-5,fy-5,10,10);
    x.strokeStyle="rgba(255,167,38,0.5)";x.lineWidth=1.5;x.setLineDash([4,4]);
    x.beginPath();x.moveTo(sx+ps/2,sy+ps/2);x.lineTo(fx,fy);x.stroke();x.setLineDash([]);
    if(fly>=1){
      x.fillStyle="#66bb6a";x.font="11px Arial";x.fillText("placed",sp[0]+8,sp[1]+4);
    }
  }
  if(t%cycle===cycle-1){idx=(idx+1)%spots.length;}
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 12
    {
        "keywords": [
            "confusion matrix", "confused", "misclassified", "errors",
            "river highway", "mistakes", "misclassification", "off-diagonal",
        ],
        "tab_title": "Reading Confusion Matrices",
        "answer": (
            "A confusion matrix is a grid that lays bare exactly where the model gets "
            "things right and where it gets confused. The rows are the true classes "
            "and the columns are what the model predicted, so a cell tells you how "
            "many images of a given true class were called a given predicted class. "
            "The diagonal — where true equals predicted — is where you want all the "
            "weight, because those are the correct calls. Everything off the diagonal "
            "is a mistake, and the brightest off-diagonal cells reveal the model's "
            "systematic confusions. In this EuroSAT project the worst off-diagonal "
            "cell is River truth, Highway prediction, with 81 River images "
            "misclassified as Highway — both are long, thin, dark ribbons from above. "
            "The next-worst is PermanentCrop truth, HerbaceousVegetation prediction, "
            "with 49 mix-ups, since green farmland and green wild vegetation look alike "
            "from 64x64 satellite tiles. Reading those cells tells you precisely which "
            "classes to gather more data for. The animation fills a small grid cell by "
            "cell: the diagonal glows green while the River-to-Highway off-diagonal "
            "cell pulses red with its 81 count."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Rows = truth, cols = predicted; green diagonal is correct, red cell is River->Highway (81)</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var labels=["River","Highway","Crop","Veg"];
var n=4, cell=52, ox=140, oy=70;
// row = truth, col = predicted; diagonal high, one off-diagonal hot
var counts=[[210,81,3,6],[12,240,4,9],[5,7,200,49],[4,8,40,205]];
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var total=n*n;
  var filled=Math.floor((t%200)/200*(total+4)); // sweep fill then hold
  if(filled>total)filled=total;
  x.fillStyle="#9aa5b1";x.font="12px Arial";
  x.fillText("predicted ->",ox+30,oy-26);
  x.save();x.translate(ox-44,oy+2*cell);x.rotate(-Math.PI/2);
  x.fillText("truth ->",-20,0);x.restore();
  // column labels
  x.font="10px Arial";x.fillStyle="#9aa5b1";
  for(var c=0;c<n;c++){x.fillText(labels[c],ox+c*cell+6,oy-8);}
  for(var r=0;r<n;r++){
    x.fillStyle="#9aa5b1";x.font="10px Arial";
    x.fillText(labels[r],ox-44,oy+r*cell+cell/2+4);
    for(var cc=0;cc<n;cc++){
      var idx=r*n+cc;
      var on = idx<filled;
      var bx=ox+cc*cell, by=oy+r*cell;
      var diag=(r===cc);
      if(!on){
        x.fillStyle="#1b2330";
      } else if(diag){
        x.fillStyle="#66bb6a";
      } else if(r===0 && cc===1){
        var pulse=0.5+0.5*Math.sin(t*0.12);
        x.fillStyle="rgba(255,107,107,"+(0.45+0.55*pulse)+")";
      } else {
        // intensity by count
        var v=counts[r][cc];
        var a=Math.min(0.7,v/120);
        x.fillStyle="rgba(255,167,38,"+(0.12+a)+")";
      }
      x.fillRect(bx+1,by+1,cell-2,cell-2);
      if(on){
        x.fillStyle=(diag||(r===0&&cc===1))?"#0f1419":"#e8edf2";
        x.font="bold 11px Arial";
        x.fillText(counts[r][cc],bx+cell/2-10,by+cell/2+4);
      }
    }
  }
  // highlight the hot cell label
  if(filled>=2){
    var hb_x=ox+1*cell, hb_y=oy+0*cell;
    var pw=2+2*Math.abs(Math.sin(t*0.12));
    x.strokeStyle="#ff6b6b";x.lineWidth=pw;
    x.strokeRect(hb_x,hb_y,cell,cell);
    x.fillStyle="#ff6b6b";x.font="bold 12px Arial";
    x.fillText("River->Highway: 81",ox+n*cell+10,oy+cell/2);
  }
  // legend
  x.fillStyle="#66bb6a";x.fillRect(ox,oy+n*cell+16,12,12);
  x.fillStyle="#e8edf2";x.font="11px Arial";x.fillText("correct (diagonal)",ox+16,oy+n*cell+26);
  x.fillStyle="#ff6b6b";x.fillRect(ox+150,oy+n*cell+16,12,12);
  x.fillStyle="#e8edf2";x.fillText("top confusion",ox+166,oy+n*cell+26);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 13
    {
        "keywords": [
            "precision", "recall", "f1", "false positive", "false negative",
            "true positive", "precision vs recall", "precision and recall",
        ],
        "tab_title": "Precision vs Recall",
        "answer": (
            "Precision and recall are two different ways of being right, and they pull "
            "in opposite directions. Take the River class. Precision asks: of all the "
            "images the model labeled River, what fraction were actually rivers? It "
            "punishes false positives — calling a highway a river. Recall asks: of all "
            "the images that truly were rivers, what fraction did the model catch? It "
            "punishes false negatives — missing a real river. If the model is timid and "
            "only flags the most obvious wide rivers, precision is high but recall is "
            "low because it misses the narrow ones. If it is trigger-happy and calls "
            "every dark ribbon a river, recall climbs but precision drops as highways "
            "sneak in — and remember 81 real rivers were called highways in this "
            "project, which is exactly a River recall problem. The F1 score is the "
            "harmonic mean that balances the two into one number. The animation shows "
            "true rivers as blue dots and everything else as gray; a movable 'predicted "
            "river' circle grows and shrinks while precision and recall bars update "
            "live, so you can watch the trade-off happen."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">Blue = true rivers; the predicted-river circle grows/shrinks, trading precision for recall</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
// fixed scatter: some true rivers (blue) clustered, gray others spread
var dots=[];
var seed=777;
function rng(){seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;}
var cxC=160, cyC=160; // cluster center for true rivers
for(var i=0;i<26;i++){
  // true rivers near center
  var ang=rng()*6.28, rad=rng()*70;
  dots.push({x:cxC+Math.cos(ang)*rad, y:cyC+Math.sin(ang)*rad, river:true});
}
for(var j=0;j<34;j++){
  dots.push({x:60+rng()*230, y:70+rng()*180, river:false});
}
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  // moving predicted-river radius: oscillate 25..120
  var rr=72+62*Math.sin(t*0.02);
  // circle
  x.strokeStyle="#ffa726";x.lineWidth=2;x.setLineDash([5,4]);
  x.beginPath();x.arc(cxC,cyC,rr,0,7);x.stroke();x.setLineDash([]);
  x.fillStyle="rgba(255,167,38,0.07)";x.beginPath();x.arc(cxC,cyC,rr,0,7);x.fill();
  x.fillStyle="#ffa726";x.font="11px Arial";x.fillText("predicted river",cxC-36,cyC-rr-8);
  var tp=0,fp=0,fn=0;
  for(var k=0;k<dots.length;k++){
    var d=dots[k];
    var dx=d.x-cxC, dy=d.y-cyC;
    var inside=(dx*dx+dy*dy)<=rr*rr;
    if(d.river && inside)tp++;
    if(!d.river && inside)fp++;
    if(d.river && !inside)fn++;
    x.beginPath();x.arc(d.x,d.y,5,0,7);
    x.fillStyle=d.river?"#4fc3f7":"#5a6675";
    x.fill();
    if(inside){x.strokeStyle="#ffa726";x.lineWidth=1.5;x.stroke();}
  }
  var prec=(tp+fp)>0?tp/(tp+fp):0;
  var rec=(tp+fn)>0?tp/(tp+fn):0;
  // bars
  var bx=330, bw=150, bh=22;
  x.fillStyle="#9aa5b1";x.font="11px Arial";
  x.fillText("Precision = TP/(TP+FP)",bx,70);
  x.fillStyle="#2a3340";x.fillRect(bx,78,bw,bh);
  x.fillStyle="#66bb6a";x.fillRect(bx,78,bw*prec,bh);
  x.fillStyle="#e8edf2";x.font="bold 12px Arial";x.fillText((prec*100).toFixed(0)+"%",bx+bw+8,95);
  x.fillStyle="#9aa5b1";x.font="11px Arial";
  x.fillText("Recall = TP/(TP+FN)",bx,128);
  x.fillStyle="#2a3340";x.fillRect(bx,136,bw,bh);
  x.fillStyle="#4fc3f7";x.fillRect(bx,136,bw*rec,bh);
  x.fillStyle="#e8edf2";x.font="bold 12px Arial";x.fillText((rec*100).toFixed(0)+"%",bx+bw+8,153);
  var f1=(prec+rec)>0?2*prec*rec/(prec+rec):0;
  x.fillStyle="#ffa726";x.font="bold 12px Arial";x.fillText("F1 = "+(f1*100).toFixed(0)+"%",bx,190);
  // counts
  x.fillStyle="#9aa5b1";x.font="11px Arial";
  x.fillText("TP="+tp+"  FP="+fp+"  FN="+fn,bx,214);
  // legend
  x.fillStyle="#4fc3f7";x.beginPath();x.arc(bx+6,238,5,0,7);x.fill();
  x.fillStyle="#e8edf2";x.fillText("true river",bx+16,242);
  x.fillStyle="#5a6675";x.beginPath();x.arc(bx+6,262,5,0,7);x.fill();
  x.fillStyle="#e8edf2";x.fillText("not a river",bx+16,266);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 14
    {
        "keywords": [
            "tensor", "shape", "dimensions", "3x64x64", "array",
            "tensors", "dimension", "n-dimensional",
        ],
        "tab_title": "What Is a Tensor",
        "answer": (
            "A tensor is just a container for numbers, and its rank is how many "
            "directions you can index along. A scalar is a single number, rank 0. A "
            "vector is a row of numbers, rank 1, like a list of 10 class scores. A "
            "matrix is a grid, rank 2, with rows and columns. A tensor is the general "
            "term for any of these and especially for stacks of higher rank. In this "
            "EuroSAT project a single image is a rank-3 tensor of shape 3 by 64 by 64: "
            "3 color channels (red, green, blue), each a 64 by 64 grid of pixels. When "
            "we train, we do not feed images one at a time; we stack a whole batch "
            "together into a rank-4 tensor of shape 128 by 3 by 64 by 64, meaning 128 "
            "images, each with 3 channels of 64 by 64 pixels. The shape is the model's "
            "native language — every layer transforms one tensor shape into another. "
            "The animation builds it up: a single number stretches into a row, the row "
            "stacks into a grid, the grid becomes a 3-layer colored cube (the image), "
            "and finally many cubes line up into a batch, with a shape label at each "
            "stage."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">scalar -> vector -> matrix -> 3x64x64 image cube -> 128x3x64x64 batch</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var stageNames=["scalar  ()","vector  (10,)","matrix  (64,64)","tensor  (3,64,64)","batch  (128,3,64,64)"];
var stageDesc=["one number","a row of numbers","rows x columns","3 channels of 64x64","128 images stacked"];
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var perStage=110;
  var stage=Math.floor((t%(perStage*5))/perStage);
  var local=((t%(perStage*5))%perStage)/perStage; // 0..1 within stage
  var cxp=260, cyp=150;
  x.textAlign="center";
  if(stage===0){
    x.fillStyle="#4fc3f7";x.fillRect(cxp-22,cyp-22,44,44);
    x.fillStyle="#0f1419";x.font="bold 16px Arial";x.fillText("7",cxp,cyp+6);
  } else if(stage===1){
    var nshow=Math.min(10,Math.floor(local*12)+1);
    for(var i=0;i<nshow;i++){
      x.fillStyle="#4fc3f7";x.fillRect(cxp-110+i*22,cyp-16,20,32);
      x.fillStyle="#0f1419";x.font="bold 11px Arial";x.fillText(i,cxp-110+i*22+10,cyp+4);
    }
  } else if(stage===2){
    var rows=Math.min(6,Math.floor(local*8)+1);
    for(var r=0;r<rows;r++)for(var c=0;c<8;c++){
      x.fillStyle=((r+c)%2)?"#4fc3f7":"#3a8fc0";
      x.fillRect(cxp-88+c*22,cyp-66+r*22,20,20);
    }
    x.fillStyle="#9aa5b1";x.font="10px Arial";x.fillText("(64 x 64 shown small)",cxp,cyp+70);
  } else if(stage===3){
    // 3-layer colored cube
    var cols=["rgba(229,84,84,0.9)","rgba(102,187,106,0.9)","rgba(79,195,247,0.9)"];
    var labs=["R","G","B"];
    var depth=Math.min(3,Math.floor(local*4)+1);
    for(var d=depth-1;d>=0;d--){
      var off=d*18;
      x.fillStyle=cols[d];
      x.fillRect(cxp-60+off,cyp-50-off,90,90);
      x.strokeStyle="#0f1419";x.lineWidth=1.5;x.strokeRect(cxp-60+off,cyp-50-off,90,90);
      x.fillStyle="#0f1419";x.font="bold 14px Arial";x.fillText(labs[d],cxp-44+off,cyp-30-off);
    }
    x.fillStyle="#9aa5b1";x.font="10px Arial";x.fillText("one image = 3 x 64 x 64",cxp,cyp+62);
  } else {
    // batch of cubes
    var ncubes=Math.min(8,Math.floor(local*10)+1);
    for(var b=0;b<ncubes;b++){
      var bx=70+b*52;
      var cols2=["#e55454","#66bb6a","#4fc3f7"];
      for(var dd=2;dd>=0;dd--){
        var o=dd*7;
        x.fillStyle=cols2[dd];
        x.fillRect(bx+o,cyp-30-o,38,38);
        x.strokeStyle="#0f1419";x.lineWidth=1;x.strokeRect(bx+o,cyp-30-o,38,38);
      }
    }
    x.fillStyle="#9aa5b1";x.font="10px Arial";x.fillText("... 128 images in a batch",cxp,cyp+70);
  }
  // stage label
  x.fillStyle="#ffa726";x.font="bold 16px Courier";x.fillText(stageNames[stage],cxp,46);
  x.fillStyle="#66bb6a";x.font="12px Arial";x.fillText(stageDesc[stage],cxp,72);
  // progress dots
  for(var s=0;s<5;s++){
    x.fillStyle=(s===stage)?"#ffa726":"#3a4656";
    x.beginPath();x.arc(cxp-40+s*20,280,5,0,7);x.fill();
  }
  x.textAlign="left";
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },

    # ----------------------------------------------------------------- 15
    {
        "keywords": [
            "augmentation", "augment", "flip", "rotate", "more data",
            "data augmentation", "augmenting", "augmented", "rotation",
        ],
        "tab_title": "Data Augmentation",
        "answer": (
            "Data augmentation stretches a limited dataset by creating new, "
            "label-preserving variants of the images you already have. You flip, "
            "rotate, shift, or recolor each picture slightly, and because the label "
            "does not change, the model sees effectively more examples and learns to "
            "be robust to those variations. Satellite imagery is an especially natural "
            "fit because the view is straight down: a forest is still a forest whether "
            "you flip it left-to-right or rotate it 90 degrees, since there is no "
            "fixed 'up' the way there is in a photo of a person. That makes flips and "
            "rotations free, safe ways to multiply the 27,000 EuroSAT patches. Honest "
            "note: this project's baseline did NOT use augmentation — it relied on the "
            "frozen ResNet18 backbone alone — so adding flips and rotations is one of "
            "the clearest, easiest upgrades available to push past 85.1% accuracy. The "
            "animation shows one satellite patch in the center spawning flipped and "
            "rotated copies that orbit around it, each labeled with the transform that "
            "made it."
        ),
        "visual_html": r'''<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#0f1419;color:#e8edf2;font-family:Segoe UI,Arial,sans-serif}
canvas{display:block;margin:0 auto}
.cap{text-align:center;font-size:12px;padding:6px;color:#9aa5b1}
</style></head><body>
<canvas id="c" width="520" height="300"></canvas>
<div class="cap">One patch spawns label-preserving flips & rotations (not used in this baseline)</div>
<script>
var cv=document.getElementById("c"),x=cv.getContext("2d");
var W=520,H=300,t=0;
var cxp=260,cyp=150;
var variants=[
  {lab:"flip H",hflip:true,vflip:false,rot:0},
  {lab:"flip V",hflip:false,vflip:true,rot:0},
  {lab:"rot 90",hflip:false,vflip:false,rot:Math.PI/2},
  {lab:"rot 180",hflip:false,vflip:false,rot:Math.PI},
  {lab:"rot 270",hflip:false,vflip:false,rot:3*Math.PI/2},
  {lab:"flip+rot",hflip:true,vflip:false,rot:Math.PI/2}
];
function drawPatch(cx,cy,size,v,alpha){
  x.save();
  x.globalAlpha=alpha;
  x.translate(cx,cy);
  if(v){
    x.rotate(v.rot);
    x.scale(v.hflip?-1:1, v.vflip?-1:1);
  }
  var s=size;
  // a "satellite-ish" patch: green field with a diagonal blue river + brown corner
  x.fillStyle="#3f7a3f";x.fillRect(-s/2,-s/2,s,s);
  x.fillStyle="#4fc3f7";x.beginPath();
  x.moveTo(-s/2,-s/4);x.lineTo(-s/4,-s/2);x.lineTo(s/2,s/4);x.lineTo(s/4,s/2);x.closePath();x.fill();
  x.fillStyle="#a9763f";x.fillRect(s/2-s/3,-s/2,s/3,s/3);
  x.strokeStyle="#0f1419";x.lineWidth=1.5;x.strokeRect(-s/2,-s/2,s,s);
  x.restore();
}
function frame(){
  x.fillStyle="#0f1419";x.fillRect(0,0,W,H);
  var orbitR=100;
  var ph=(t%600)/600; // slow reveal
  var shown=Math.min(variants.length,Math.floor(t/55)%(variants.length+3));
  // orbiting variants
  for(var i=0;i<variants.length;i++){
    var a=t*0.012 + i*(6.283/variants.length);
    var ox=cxp+Math.cos(a)*orbitR;
    var oy=cyp+Math.sin(a)*orbitR*0.62;
    var appear=i<shown?1:0.12;
    // connector line
    x.strokeStyle="rgba(255,167,38,"+(0.15*appear)+")";x.lineWidth=1;
    x.beginPath();x.moveTo(cxp,cyp);x.lineTo(ox,oy);x.stroke();
    drawPatch(ox,oy,46,variants[i],appear);
    if(i<shown){
      x.fillStyle="#ffa726";x.font="10px Arial";x.textAlign="center";
      x.fillText(variants[i].lab,ox,oy+38);x.textAlign="left";
    }
  }
  // original in center
  drawPatch(cxp,cyp,64,null,1);
  x.fillStyle="#66bb6a";x.font="bold 11px Arial";x.textAlign="center";
  x.fillText("original",cxp,cyp+50);x.textAlign="left";
  // labels
  x.fillStyle="#9aa5b1";x.font="11px Arial";
  x.fillText("label stays 'Forest' through every transform",20,28);
  x.fillStyle="#ffa726";x.font="11px Arial";
  x.fillText("baseline: NOT used (easy upgrade)",20,290);
  t++;requestAnimationFrame(frame);
}
frame();
</script></body></html>''',
    },
]


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def lookup(question):
    """Return the best KB entry, or None unless the match is CONFIDENT.

    Fixes for irrelevant matches: keywords match on word boundaries (so
    'lr' can't fire inside 'already'), and a single weak hit is rejected —
    we accept only >=2 distinct keyword hits, or one hit on a specific
    keyword (a multi-word phrase or a long term). Unsure -> None -> the
    live AI answers instead of a wrong instant card.
    """
    if not question:
        return None
    q = question.lower()
    best, best_score = None, 0
    for entry in KB:
        hits, specific = 0, False
        for kw in entry["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", q):
                hits += 1
                if " " in kw or len(kw) >= 8:
                    specific = True
        if hits and (hits >= 2 or specific):
            score = hits + (1 if specific else 0)
            if score > best_score:
                best_score, best = score, entry
    return best
