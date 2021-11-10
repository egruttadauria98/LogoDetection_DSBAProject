def Model_Setting():
    MODELS_CONFIGURATION = {
        'efficientdet-d0': {
            'model_name': 'efficientdet_d0_coco17_tpu-32',
            'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
            'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
            'batch_size': 15
        },
        'centernet_hg104':{
            'model_name': 'centernet_hg104_512x512_coco17_tpu-8',
            'base_pipeline_file': 'center_net_deepmac_512x512_voc_only_tpu-32.config',
            'pretrained_checkpoint': 'centernet_hg104_512x512_coco17_tpu-8.tar.gz',
            'batch_size': 15
        },
        'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8':{
            'model_name': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8',
            'base_pipeline_file': 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config',
            'pretrained_checkpoint': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz',
            'batch_size': 2
        },
        'ssd_resnet152_v1':{
            'model_name': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8',
            'base_pipeline_file': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.config',
            'pretrained_checkpoint': 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz',
            'batch_size': 2
        },
        


    }
    
    return MODELS_CONFIGURATION