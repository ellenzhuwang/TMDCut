from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from TMDCut import TMDCut_Segmentation

import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np

img = Image.open('./voc2017val/2007_000464.jpg')
name_list = ['ground', 'sea','cow','stone']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list) - 1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()

img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')


model = TMDCut_Segmentation(clip_type='openai', model_type='ViT-B-16', vfm_model='dino',
                              name_path='./configs/my_name.txt')

seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

colors = sns.hls_palette(len(np.unique(seg_pred)), h=0.9)
cmap = ListedColormap(colors)

plt.imshow(img)
plt.imshow(seg_pred, cmap=cmap, alpha=0.8)
plt.axis('off')
plt.tight_layout()
plt.savefig('seg_voc.png', bbox_inches='tight')

