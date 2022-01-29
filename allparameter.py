class allparameter:
    def __init__(self):

        self.weights='./weights/yolov4-csp-s-leaky.weights'
        #'initial weights path'
        self.cfg='./cfg/yolov4-csp-s-leaky.cfg'
        #'model.yaml path'
        self.data='datasets/obj.yaml'
        #'data.yaml path'
        self.hyp='data/hyp.scratch.yaml'
        #'hyperparameters path'
        self.epochs=3
        self.batch_size=20
        self.total_batch_size = 20
        #'total batch size for all GPUs'
        self.img_size=[448,448]
        self.rect=False 
        #'rectangular training')
        self.resume=False
        #'resume most recent training'
        self.nosave=False
        #'only save final checkpoint')
        self.notest=False
        #help='only test final epoch')
        self.noautoanchor=False
        #help='disable autoanchor check')
        self.evolve=False
        #help='evolve hyperparameters')
        self.bucket=''
        #'gsutil bucket')
        self.cache_images=False
        #help='cache images for faster training')
        self.image_weights=False
        #help='use weighted image selection for training')
        self.name=''
        #help='renames results.txt to results_name.txt if supplied')
        self.device=''
        #help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.multi_scale=False
        #help='vary img-size +/- 50%%')
        self.single_cls=False
        #help='train as single-class dataset')
        self.adam=False
        #help='use torch.optim.Adam() optimizer')
        self.sync_bn=False 
        #help='use SyncBatchNorm, only available in DDP mode')
        self.local_rank=-1
        #help='DDP parameter, do not modify')
        self.logdir='runs/'
        #help='logging directory')
        self.workers=0
        #help='maximum number of dataloader workers')
        self.world_size=1
        self.project = 'runs/train'
        self.name = 'exp'
        self.exist_ok = True
        
        
 