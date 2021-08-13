# %%
%load_ext autoreload
%autoreload 2

# standard ecosystem
import os, sys, time, copy
import numpy as np
from pathlib import Path
from PIL import Image

prefix = "../"
sys.path.append(f"{prefix}src")

# torchvision
from HummingBirdLoader import HummingBirdLoader, Denormalize
from torchvision import transforms

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

hub_dir = Path(f"/data/shared/hummingbird-classifier/models/").resolve()
torch.hub.set_dir(hub_dir)

print(f"current torch hub directory: {torch.hub.get_dir()}")
# %% # %%
BSIZE = 32
set_type = "annotated_videos"  # "balanced", "more_negatives", "same_camera"
dir_dict_trn = {
    "negatives": Path(f"{prefix}data/{set_type}/training_set/class_0"),
    "positives": Path(f"{prefix}data/{set_type}/training_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_val = {
    "negatives": Path(f"{prefix}data/{set_type}/validation_set/class_0"),
    "positives": Path(f"{prefix}data/{set_type}/validation_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

dir_dict_tst = {
    "negatives": Path(f"{prefix}data/{set_type}/test_set/class_0"),
    "positives": Path(f"{prefix}data/{set_type}/test_set/class_1"),
    "meta_data": Path(f"{prefix}data/positives_verified.csv"),
}

augment_tr = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        # transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.RandomEqualize(),
        # transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.5, hue=0.1),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

augment_ts = transforms.Compose(
    [
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),  # AT LEAST 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

trn_hummingbirds = HummingBirdLoader(
    dir_dict_trn, learning_set="trn", ls_inds=[], transforms=augment_tr
)

val_hummingbirds = HummingBirdLoader(
    dir_dict_val, learning_set="val", ls_inds=[], transforms=augment_ts
)

tst_hummingbirds = HummingBirdLoader(
    dir_dict_tst, learning_set="tst", ls_inds=[], transforms=augment_ts
)

trn_loader = DataLoader(
    trn_hummingbirds, batch_size=BSIZE, shuffle=True, drop_last=True
)

val_loader = DataLoader(val_hummingbirds, shuffle=False, batch_size=BSIZE)
tst_loader = DataLoader(tst_hummingbirds, shuffle=False, batch_size=BSIZE)
# %% 
# architecture = f"ResNet50_same_camera_jitter_augmentation_20210808_224x"
architecture = f"ResNet50_annotated_videos_jitter_augmentation_20210809_224x"
model_folder = Path(f"{hub_dir}/{architecture}/")

probs = np.load(model_folder / "predictions_proba.npy", allow_pickle=True)
gt = np.load(model_folder / "predictions_gt.npy", allow_pickle=True)
yhat = np.load(model_folder / "predictions_yhat.npy", allow_pickle=True)

# %%
cl = 0
tst_sub = probs[gt==cl,:]
labs = gt[gt==cl]
sind = np.argsort(1-tst_sub[:,1])
# possort = tst_positives[ii]
xfiles = tst_hummingbirds.img_paths[gt==cl]
xnames = tst_hummingbirds.img_paths[gt==cl]
# xsort = xsort[ii]

c = 0
for i, ss in enumerate(sind[5:]): 
    if "FH303" not in str(xfiles[ss]): 
        continue
    # elif "FH107" in str(xfiles[ss]): 
    #     continue

    with open(xfiles[ss], "rb") as f:
            img = Image.open(f).convert("RGB")

    plt.figure()
    print(f"{c} : {ss} : LABEL: {int(labs[ss])}, 0: {tst_sub[ss,0]:.4f} - 1: {tst_sub[ss,1]:.4f}\n"
              f"FILE: {xnames[ss].name}")
    plt.title(f"Hummingbird probability: {tst_sub[ss,1]:.2f}, GT: {int(labs[ss])}")
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    if c >= 10: 
        break
    
    c += 1

# %%

if 0:
    entropy = -np.sum(np.log10(probs+1e-8) * (probs+1e-8), axis=1)

    if 0:
        plt.figure(figsize=(15,5))
        # plt.hist(entropy, bins=50, density=True)#, histtype="step")
        plt.scatter(range(len(gt)), entropy)
        plt.scatter(range(len(gt)), gt*0.302)

    sind = np.argsort(-entropy)
    # possort = tst_positives[ii]
    xfiles = tst_hummingbirds.img_paths
    xnames = tst_hummingbirds.img_paths
    # xsort = xsort[ii]

    c = 0
    for i, ss in enumerate(sind[:]): 
        with open(xfiles[ss], "rb") as f:
                img = Image.open(f).convert("RGB")

        plt.figure()
        print(f"{c} : LABEL: {int(gt[ss])}, 0: {entropy[ss]:.4f}\n"
                f"FILE: {xnames[ss].name}")
        plt.title(f"Hummingbird entropy: {entropy[ss]:.2f}, GT: {int(gt[ss])}, p0: {probs[ss,0]:.4f} - p1: {probs[ss,1]:.4f}")
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        if c > 5: 
            break
        
        c += 1
# %%

if 0: 
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.layer4[1].register_forward_hook(get_activation("layer4"))

    # Redo with random shuffle so we get a mix of classes
    tst_loader_sh = DataLoader(tst_hummingbirds, shuffle=True, batch_size=BSIZE)

    # visualize_model(model, tst_loader_sh, device="cpu", num_images=BSIZE, denormalize=denorm, save_folder=model_folder / "example_figs")

    denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for batch, (xb,yb) in enumerate(tst_loader_sh):
        for i, (x,y) in enumerate(zip(xb,yb)):
            # print(y)
            # x, y = p
            # x = denorm(x)#.permute((1,2,0))
            out = model(x[None, ...])#.to("cuda"))
            out = out.cpu()
            print(y.numpy(), out.max(dim=1)[1].numpy()[0], nn.Softmax(dim=-1)(out[0]).detach().numpy())

            # print(out)
            plt.figure()
            # plt.title(f"y {}, gt {}, yhat {}, ")
            plt.imshow(denorm(x).permute((1,2,0)))
            if i == 0: 
                break

        break

# output = model(img)
# %%
if 0:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans 

    act = torch.squeeze(activation['layer4']).cpu().permute((1,2,0)).numpy()#.max(dim=2)[1]

    act_r = act.reshape(-1,2048)
    pca = PCA(n_components=12, whiten=True)
    x_tr = pca.fit_transform(act_r)

    clu = KMeans(n_clusters=10)
    clu = clu.fit(x_tr)
    cind = clu.labels_.reshape(7,7)#[:,::-1]

    x_2d = x_tr.reshape(7,7,-1)#[:,::-1,:]
    x_2d = np.uint8(255 * (x_2d - np.min(x_2d)) / (np.max(x_2d) - np.min(x_2d)))

    c = 0
    f, a = plt.subplots(1,2)
    a[0].imshow(x.cpu().permute((1,2,0)))
    a[1].imshow(x_2d[:,:,c:c+3])

    c = 3
    f, a = plt.subplots(1,2)
    a[0].imshow(x.cpu().permute((1,2,0)))
    a[1].imshow(x_2d[:,:,c:c+3])

    c = 6
    f, a = plt.subplots(1,2)
    a[0].imshow(x.cpu().permute((1,2,0)))
    a[1].imshow(x_2d[:,:,c:c+3])

    c = 9
    f, a = plt.subplots(1,2)
    a[0].imshow(x.cpu().permute((1,2,0)))
    a[1].imshow(x_2d[:,:,c:c+3])

    f, a = plt.subplots(1,2)
    a[0].imshow(x.cpu().permute((1,2,0)))
    a[1].imshow(cind)
    # %%
    from matplotlib import colors
    c = 0
    mag = np.sqrt(np.power(x_tr[:,c],2) + np.power(x_tr[:,c+1],2)).reshape(-1,1)
    angle = np.arctan2(x_tr[:,c+1], x_tr[:,c]) 

    # ax[0,1].scatter(X_embedded[:,0], X_embedded[:,1],c=angle,s=2,cmap=plt.cm.hsv)

    angle = (angle-np.min(angle)) / (np.max(angle) - np.min(angle))
    mag = np.sqrt(mag)
    mag = (mag-np.min(mag)) / (np.max(mag) - np.min(mag))

    colors = colors.hsv_to_rgb(np.concatenate((angle.reshape(-1,1),  
                                mag.reshape(-1,1), #mag.reshape(-1,1),
                                mag.reshape(-1,1)), axis=1))

    # col2d = colors.reshape((-1,3))

    # ii = 0
    plt.figure()
    plt.scatter(x_tr[:,c],x_tr[:,c+1], c=colors, #df['hs'].mean(axis=1),
                    s=50, alpha=1)#, cmap=newcmap)

    f, a = plt.subplots(1,2)
    a[0].imshow(x.cpu().permute((1,2,0)))
    a[1].imshow(colors.reshape(7,7,3))

    # %%
