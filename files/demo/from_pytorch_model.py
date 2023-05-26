from segment_anything import sam_model_registry

def GetSamModel(pytorch_model:str,model_type:str="vit_h"):
    '''
    pytorch_model: 下载的segment-anything模型(pytorch格式)
    model_type: ["default", "vit_h", "vit_l", "vit_b"]
    '''
    return sam_model_registry[model_type](checkpoint=pytorch_model)