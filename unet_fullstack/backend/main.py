
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, cv2, numpy as np, torch

from pipeline import setup, state, start_training, sample_visualization, load_checkpoint, model, DEVICE, image_to_b64, compute_iou, compute_pixel_accuracy, IMAGE_SIZE, NUM_EPOCHS

app=FastAPI(title="U-Net Biomedical Segmentation API", version="1.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

FRONTEND=Path(__file__).resolve().parent.parent/"frontend"

@app.get("/")
def home(): return FileResponse(FRONTEND/"index.html")

@app.get("/api/health")
def health():
    return {"ok":True,"device":str(DEVICE),"cuda":torch.cuda.is_available(),
            "checkpoint":Path(__file__).resolve().parent.joinpath("unet_best.pt").exists()}

@app.post("/api/setup")
def api_setup():
    try: return setup()
    except Exception as e: raise HTTPException(500,str(e))

@app.get("/api/status")
def api_status(): return state

@app.post("/api/train")
def api_train():
    if state["status"]=="training": return {"started":False,"message":"Training already running."}
    try:
        started=start_training()
        return {"started":started,"message":"Training started in the backend."}
    except Exception as e: raise HTTPException(500,str(e))

@app.get("/api/visualization")
def api_visualization(index:int=0):
    try: return sample_visualization(index)
    except Exception as e: raise HTTPException(500,str(e))

@app.post("/api/predict")
async def api_predict(file:UploadFile=File(...)):
    data=await file.read()
    arr=np.frombuffer(data,np.uint8); img=cv2.imdecode(arr,cv2.IMREAD_GRAYSCALE)
    if img is None: raise HTTPException(400,"Could not read image.")
    if not load_checkpoint(): raise HTTPException(400,"No trained checkpoint. Train the model first.")
    original=img.copy()
    resized=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
    x=torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): pred=model(x).argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    mask=(pred*255).astype(np.uint8)
    return {"input":image_to_b64(original),"prediction":image_to_b64(mask),
            "width":int(original.shape[1]),"height":int(original.shape[0])}

@app.get("/api/metrics")
def api_metrics():
    try:
        if not Path(__file__).resolve().parent.joinpath("unet_best.pt").exists():
            return {"available":False,"message":"Train the model first."}
        load_checkpoint()
        from pipeline import _load_data
        _,val_loader=_load_data()
        vals=[]; accs=[]
        with torch.no_grad():
            for images,masks,_ in val_loader:
                logits=model(images.to(DEVICE)); pred=logits.argmax(1).cpu()
                vals.append(compute_iou(pred,masks)); accs.append(compute_pixel_accuracy(pred,masks))
        return {"available":True,"mean_iou":float(np.mean(vals)),"std_iou":float(np.std(vals)),
                "mean_pixel_accuracy":float(np.mean(accs)),"std_pixel_accuracy":float(np.std(accs))}
    except Exception as e: raise HTTPException(500,str(e))

@app.get("/api/notebook-info")
def notebook_info():
    return {"dataset":"ISBI 2012 EM Segmentation Challenge","images":30,"split":"24 train / 6 validation",
            "model":"UNetPadded (default) / UNetOriginal available in backend",
            "input_size":"256x256","classes":2,"base_channels":64,
            "loss":"Weighted pixel-wise cross entropy","w0":10,"sigma":5,
            "augmentation":"Shift/scale/rotate + flips + ElasticTransform + brightness/contrast + Gaussian noise",
            "optimizer":"SGD","learning_rate":1e-3,"momentum":0.99,"batch_size":2,"epochs":NUM_EPOCHS,
            "metrics":["IoU/Jaccard","Pixel Accuracy"]}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
