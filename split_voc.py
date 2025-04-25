import os
import shutil
from pathlib import Path
import random

vocpath = Path("./datasets/VOC/val")
imagelist = os.listdir(vocpath/"images")

# select randomly 10%
newvocpath1 = Path("./datasets/VOC/val1")
newvocpath2 = Path("./datasets/VOC/val2")
for image in imagelist:
    randnum = random.randint(1, 100)
    if randnum <= 10:
        shutil.copy(vocpath/"images"/image, newvocpath1/"images"/image)
        shutil.copy(vocpath/"labels"/image.replace(".jpg", ".txt"), newvocpath1/"labels"/image.replace(".jpg", ".txt"))
    else:
        shutil.copy(vocpath/"images"/image, newvocpath2/"images"/image)
        shutil.copy(vocpath/"labels"/image.replace(".jpg", ".txt"), newvocpath2/"labels"/image.replace(".jpg", ".txt"))   
    
