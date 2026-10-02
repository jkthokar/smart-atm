# Dataset

Face images are personal data, so they are **not** stored in this repository.
Build your own dataset with the webcam:

```bash
python collect_data.py --person alice     # hold G (face visible) or D (face covered)
python collect_data.py --person bob      # repeat for each participant
python split_dataset.py --by-person      # 80% train / 20% test, split by person
```

Resulting layout:

```
dataset/
├── raw/{access_granted,access_denied}/     all collected images
├── train/{access_granted,access_denied}/   used by train.py
└── test/{access_granted,access_denied}/    used only by evaluate.py
```

**Tips for a good dataset**

- Several people. Ask every participant for consent before recording.
- Different lighting (day, night, backlight), distances and head angles.
- Many kinds of covering: surgical mask, KF94, helmet, cap pulled low, hood, scarf, sunglasses with mask.
- Visible faces with glasses, hats that do not hide the face, and different hairstyles, so the model does not learn "any accessory means deny".
- Roughly the same number of images in both classes.
