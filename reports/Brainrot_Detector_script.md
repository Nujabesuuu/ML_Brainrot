# Brainrot Detector: 4-minute script

Target: about 40 seconds per slide. Roughly 330 spoken words take 3.5 minutes with pauses, which leaves a buffer under 4 minutes.

## 1 · Title (0:00–0:30)
Hi, we are Yevhenii and Heorhii. Our question: can a neural network tell Italian brainrot memes, these AI-generated animal-object hybrids, from everything else?
The short answer: yes against ordinary photos, but not yet against other AI-generated art. Here is how we got there.

## 2 · The data (0:30–1:10)
We collected almost ten thousand images: about five thousand brainrot, four and a half thousand photos, and five hundred ordinary AI-generated pictures.
That last group is the important twist. Every brainrot image is AI-made, so if we only used photos, the model could cheat by learning "AI-generated means brainrot". The ordinary AI art keeps it honest.

## 3 · What we built (1:10–1:50)
We trained four models of increasing strength. A coin flip scores 0.5. Colour alone already gets 0.77, because brainrot is very saturated. A pretrained image network jumps to 0.97, and after fine-tuning it on our data we reach 0.986, close to perfect.
All scores come from a test set the models never saw during training.

## 4 · Results (1:50–2:30)
Broken down by image type, the picture is clear. Only 1.5% of photos are wrongly called brainrot, and we miss under 5% of real brainrot. So as a brainrot-versus-photos filter, it works.
But almost 39% of ordinary AI images are wrongly flagged. That is the weak spot.

## 5 · Why (2:30–3:10)
Looking at the mistakes explains it. These are all normal AI pictures, and the model is almost 100% sure each one is brainrot.
They share big eyes, bright colours and a cartoon creature. The model partly learnt a shortcut: "cute AI creature equals brainrot". 29 of 75 such test images fell for it.

## 6 · The fix (3:10–3:50)
The good news: the fix is clear. When we give the model more ordinary AI images during training, the error drops from 58% to 32%, and the curve is still going down.
So today it is ready as a brainrot-versus-photos filter, and we have a demo app. The next step is to collect more ordinary AI art. Thank you!
