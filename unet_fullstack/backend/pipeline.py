
import os, random, math, time, subprocess, threading
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, label as cc_label
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
REPO_DIR = DATA_ROOT / "unet"
IMAGE_DIR = REPO_DIR / "data" / "membrane" / "train" / "image"
LABEL_DIR = REPO_DIR / "data" / "membrane" / "train" / "label"
CHECKPOINT_PATH = ROOT / "unet_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_CHANNELS, N_CLASSES, BASE_CHANNELS = 1, 2, 64
IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
MOMENTUM = 0.99
NUM_EPOCHS = 10
W0, SIGMA = 10.0, 5.0

state = {
    "status": "idle", "epoch": 0, "total_epochs": NUM_EPOCHS,
    "train_loss": None, "val_loss": None, "val_iou": None,
    "val_pixel_acc": None, "best_val_iou": None, "message": "",
    "history": {"train_loss": [], "val_loss": [], "val_iou": [], "val_pixel_acc": []},
}

def download_isbi_membrane_dataset():
    if IMAGE_DIR.is_dir() and any(IMAGE_DIR.iterdir()):
        return {"downloaded": False, "images": len(list(IMAGE_DIR.iterdir()))}
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/zhixuhao/unet.git", str(REPO_DIR)], check=True)
    return {"downloaded": True, "images": len(list(IMAGE_DIR.iterdir()))}

class ISBIMembraneDataset(Dataset):
    def __init__(self, image_dir, label_dir, indices=None):
        self.image_dir, self.label_dir = Path(image_dir), Path(label_dir)
        files = sorted(os.listdir(image_dir), key=lambda f: int(Path(f).stem))
        self.filenames = [files[i] for i in indices] if indices is not None else files
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = cv2.imread(str(self.image_dir/fname), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(self.label_dir/fname), cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)/255.0
        mask = (label > 127).astype(np.int64)
        return image, mask

def compute_class_balance_weight(mask):
    mask = mask.astype(np.float32)
    n_fg = max(float(mask.sum()), 1.0)
    n_bg = max(float(mask.size-mask.sum()), 1.0)
    w_fg = mask.size/(2.0*n_fg); w_bg = mask.size/(2.0*n_bg)
    return np.where(mask == 1, w_fg, w_bg).astype(np.float32)

def compute_boundary_weight(mask, w0=10.0, sigma=5.0):
    labeled, n_instances = cc_label(mask)
    if n_instances < 2:
        return np.zeros_like(mask, dtype=np.float32)
    H,W = mask.shape
    dist_stack = np.zeros((n_instances,H,W), dtype=np.float32)
    for k in range(1,n_instances+1):
        dist_stack[k-1] = distance_transform_edt(~(labeled == k))
    dist_sorted = np.sort(dist_stack, axis=0)
    d1, d2 = dist_sorted[0], dist_sorted[1]
    bw = w0*np.exp(-((d1+d2)**2)/(2.0*sigma**2))
    return np.where(mask == 0, bw, 0.0).astype(np.float32)

def compute_weight_map(mask, w0=10.0, sigma=5.0):
    return (compute_class_balance_weight(mask) +
            compute_boundary_weight(mask,w0,sigma)).astype(np.float32)

def get_train_augmentation(image_size=256):
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=25,
                           border_mode=cv2.BORDER_REFLECT, p=0.8),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.ElasticTransform(alpha=34, sigma=10, border_mode=cv2.BORDER_REFLECT, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(0.05,0.15), p=0.3),
    ])

def get_val_augmentation():
    return A.Compose([])

class UNetSegmentationDataset(Dataset):
    def __init__(self, base_dataset, input_size=256, target_size=256, augmentation=None,
                 w0=10.0, sigma=5.0):
        self.base_dataset=base_dataset; self.input_size=input_size
        self.target_size=target_size; self.augmentation=augmentation
        self.w0=w0; self.sigma=sigma
    def __len__(self): return len(self.base_dataset)
    def _center_crop(self, arr, out_size):
        h,w=arr.shape[:2]
        if h==out_size and w==out_size: return arr
        top=(h-out_size)//2; left=(w-out_size)//2
        return arr[top:top+out_size,left:left+out_size]
    def __getitem__(self, idx):
        image,mask=self.base_dataset[idx]
        image=cv2.resize(image,(self.input_size,self.input_size),interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize(mask.astype(np.uint8),(self.input_size,self.input_size),
                        interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if self.augmentation is not None:
            a=self.augmentation(image=image,mask=mask); image,mask=a["image"],a["mask"]
        weight=compute_weight_map(mask,self.w0,self.sigma)
        mask=self._center_crop(mask,self.target_size)
        weight=self._center_crop(weight,self.target_size)
        return (torch.from_numpy(image).float().unsqueeze(0),
                torch.from_numpy(mask).long(),
                torch.from_numpy(weight).float())

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch,padding):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=padding),nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=padding),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

def center_crop(x,target_size):
    _,_,h,w=x.shape; th,tw=target_size
    top=(h-th)//2; left=(w-tw)//2
    return x[:,:,top:top+th,left:left+tw]

class UNetOriginal(nn.Module):
    def __init__(self,in_channels=1,n_classes=2,base_channels=64):
        super().__init__(); c=base_channels
        self.enc1=DoubleConv(in_channels,c,0); self.enc2=DoubleConv(c,c*2,0)
        self.enc3=DoubleConv(c*2,c*4,0); self.enc4=DoubleConv(c*4,c*8,0)
        self.pool=nn.MaxPool2d(2,2); self.bottleneck=DoubleConv(c*8,c*16,0)
        self.up4=nn.ConvTranspose2d(c*16,c*8,2,2); self.dec4=DoubleConv(c*16,c*8,0)
        self.up3=nn.ConvTranspose2d(c*8,c*4,2,2); self.dec3=DoubleConv(c*8,c*4,0)
        self.up2=nn.ConvTranspose2d(c*4,c*2,2,2); self.dec2=DoubleConv(c*4,c*2,0)
        self.up1=nn.ConvTranspose2d(c*2,c,2,2); self.dec1=DoubleConv(c*2,c,0)
        self.out_conv=nn.Conv2d(c,n_classes,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2))
        e4=self.enc4(self.pool(e3)); b=self.bottleneck(self.pool(e4))
        d4=self.up4(b); d4=self.dec4(torch.cat([center_crop(e4,d4.shape[-2:]),d4],1))
        d3=self.up3(d4); d3=self.dec3(torch.cat([center_crop(e3,d3.shape[-2:]),d3],1))
        d2=self.up2(d3); d2=self.dec2(torch.cat([center_crop(e2,d2.shape[-2:]),d2],1))
        d1=self.up1(d2); d1=self.dec1(torch.cat([center_crop(e1,d1.shape[-2:]),d1],1))
        return self.out_conv(d1)

class UNetPadded(nn.Module):
    def __init__(self,in_channels=1,n_classes=2,base_channels=64):
        super().__init__(); c=base_channels
        self.enc1=DoubleConv(in_channels,c,1); self.enc2=DoubleConv(c,c*2,1)
        self.enc3=DoubleConv(c*2,c*4,1); self.enc4=DoubleConv(c*4,c*8,1)
        self.pool=nn.MaxPool2d(2,2); self.bottleneck=DoubleConv(c*8,c*16,1)
        self.up4=nn.ConvTranspose2d(c*16,c*8,2,2); self.dec4=DoubleConv(c*16,c*8,1)
        self.up3=nn.ConvTranspose2d(c*8,c*4,2,2); self.dec3=DoubleConv(c*8,c*4,1)
        self.up2=nn.ConvTranspose2d(c*4,c*2,2,2); self.dec2=DoubleConv(c*4,c*2,1)
        self.up1=nn.ConvTranspose2d(c*2,c,2,2); self.dec1=DoubleConv(c*2,c,1)
        self.out_conv=nn.Conv2d(c,n_classes,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2))
        e4=self.enc4(self.pool(e3)); b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([e4,self.up4(b)],1)); d3=self.dec3(torch.cat([e3,self.up3(d4)],1))
        d2=self.dec2(torch.cat([e2,self.up2(d3)],1)); d1=self.dec1(torch.cat([e1,self.up1(d2)],1))
        return self.out_conv(d1)

def weights_init_he(module):
    if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight,mode="fan_in",nonlinearity="relu")
        if module.bias is not None: nn.init.zeros_(module.bias)

class WeightedCrossEntropyLoss(nn.Module):
    def forward(self,logits,target,weight_map):
        log_probs=F.log_softmax(logits,dim=1)
        nll=F.nll_loss(log_probs,target,reduction="none")
        return (nll*weight_map).sum()/(weight_map.sum()+1e-8)

@torch.no_grad()
def compute_iou(pred,target,n_classes=2,eps=1e-8):
    ious=[]
    for cls in range(n_classes):
        pc=(pred==cls); tc=(target==cls)
        inter=(pc&tc).sum().float(); union=(pc|tc).sum().float()
        if union != 0: ious.append((inter/(union+eps)).item())
    return float(np.mean(ious)) if ious else 0.0

@torch.no_grad()
def compute_pixel_accuracy(pred,target):
    return ((pred==target).sum().float()/torch.numel(target)).item()

model = UNetPadded(IN_CHANNELS,N_CLASSES,BASE_CHANNELS).to(DEVICE)
model.apply(weights_init_he)
criterion=WeightedCrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=4)

def dataset_split():
    files=sorted(os.listdir(IMAGE_DIR),key=lambda f:int(Path(f).stem))
    idx=list(range(len(files))); random.Random(SEED).shuffle(idx)
    val=sorted(idx[:6]); train=sorted(idx[6:])
    return train,val

def setup():
    info=download_isbi_membrane_dataset()
    train_idx,val_idx=dataset_split()
    return {"device":str(DEVICE),"cuda":torch.cuda.is_available(),
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "images":len(os.listdir(IMAGE_DIR)),"train":len(train_idx),"val":len(val_idx),
            "downloaded":info["downloaded"]}

def _load_data():
    train_idx,val_idx=dataset_split()
    raw_train=ISBIMembraneDataset(IMAGE_DIR,LABEL_DIR,train_idx)
    raw_val=ISBIMembraneDataset(IMAGE_DIR,LABEL_DIR,val_idx)
    train_ds=UNetSegmentationDataset(raw_train,IMAGE_SIZE,IMAGE_SIZE,get_train_augmentation(IMAGE_SIZE),W0,SIGMA)
    val_ds=UNetSegmentationDataset(raw_val,IMAGE_SIZE,IMAGE_SIZE,get_val_augmentation(),W0,SIGMA)
    return DataLoader(train_ds,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,drop_last=True), DataLoader(val_ds,1,False,num_workers=NUM_WORKERS)

def train_one_epoch(loader):
    model.train(); running=0.0
    for images,masks,weights in loader:
        images,masks,weights=images.to(DEVICE),masks.to(DEVICE),weights.to(DEVICE)
        optimizer.zero_grad(); logits=model(images); loss=criterion(logits,masks,weights)
        loss.backward(); optimizer.step(); running += loss.item()*images.size(0)
    return running/len(loader.dataset)

@torch.no_grad()
def validate(loader):
    model.eval(); running=0.; ious=[]; accs=[]
    for images,masks,weights in loader:
        images,masks,weights=images.to(DEVICE),masks.to(DEVICE),weights.to(DEVICE)
        logits=model(images); loss=criterion(logits,masks,weights); running += loss.item()
        pred=logits.argmax(1); ious.append(compute_iou(pred,masks)); accs.append(compute_pixel_accuracy(pred,masks))
    return running/len(loader.dataset),float(np.mean(ious)),float(np.mean(accs))

def train():
    if not IMAGE_DIR.exists(): setup()
    train_loader,val_loader=_load_data()
    state.update(status="training",epoch=0,message="Training started.",history={"train_loss":[],"val_loss":[],"val_iou":[],"val_pixel_acc":[]})
    best=-1.; start=time.time()
    for epoch in range(1,NUM_EPOCHS+1):
        tl=train_one_epoch(train_loader); vl,vi,va=validate(val_loader); scheduler.step(vi)
        state.update(epoch=epoch,train_loss=tl,val_loss=vl,val_iou=vi,val_pixel_acc=va)
        for k,v in [("train_loss",tl),("val_loss",vl),("val_iou",vi),("val_pixel_acc",va)]:
            state["history"][k].append(v)
        if vi>best:
            best=vi; state["best_val_iou"]=best
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),"val_iou":vi,
                        "unet_variant":"padded"},CHECKPOINT_PATH)
        state["message"]=f"Epoch {epoch}/{NUM_EPOCHS} complete"
    state.update(status="completed",message=f"Training complete in {(time.time()-start)/60:.1f} min.")
    return state

def start_training():
    if state["status"]=="training": return False
    threading.Thread(target=train,daemon=True).start(); return True

def load_checkpoint():
    if not CHECKPOINT_PATH.exists(): return False
    ck=torch.load(CHECKPOINT_PATH,map_location=DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    state["best_val_iou"]=ck.get("val_iou")
    return True

def image_to_b64(arr):
    import base64
    ok,buf=cv2.imencode(".png",arr)
    if not ok: raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64,"+base64.b64encode(buf).decode()

@torch.no_grad()
def sample_visualization(index=0):
    if not IMAGE_DIR.exists(): setup()
    _,val_idx=dataset_split()
    raw=ISBIMembraneDataset(IMAGE_DIR,LABEL_DIR,val_idx)
    ds=UNetSegmentationDataset(raw,IMAGE_SIZE,IMAGE_SIZE,get_val_augmentation(),W0,SIGMA)
    image,mask,weight=ds[index%len(ds)]
    loaded=load_checkpoint()
    pred=None
    if loaded:
        logits=model(image.unsqueeze(0).to(DEVICE)); pred=logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)*255
    return {
        "input":image_to_b64((image.squeeze().numpy()*255).astype(np.uint8)),
        "ground_truth":image_to_b64((mask.numpy()*255).astype(np.uint8)),
        "weight_map":image_to_b64(cv2.normalize(weight.numpy(),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)),
        "prediction":image_to_b64(pred) if pred is not None else None,
        "metrics": {"iou": compute_iou(torch.from_numpy(pred[None]//255),mask[None]) if pred is not None else None,
                    "pixel_accuracy": compute_pixel_accuracy(torch.from_numpy(pred[None]//255),mask[None]) if pred is not None else None}
    }
