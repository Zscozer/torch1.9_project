# #hugging face 预训练模型简单使用
# from transformers import ViTFeatureExtractor, ViTModel
# from PIL import Image
# import requests
#
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
#
# # print(image)
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
# # print(feature_extractor)
# # print(model)
# inputs = feature_extractor(images=image, return_tensors="pt")
#
# outputs = model(**inputs)
# # print(outputs)
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)
# print(outputs.pooler_output.shape)

import torch
a = torch.rand(8,4800,dtype=float)
i = a.shape[0]
print(i)