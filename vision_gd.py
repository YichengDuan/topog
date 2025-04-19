from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests


class ObjectExtractor:
    def __init__(self, model_name="facebook/detr-resnet-50", threshold=0.9):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm").to(self.device)
        self.threshold = threshold

    def extract_objects(self, img_vector):
        # single-image wrapper
        return self.extract_batch([img_vector])[0]

    def extract_batch(self, img_vectors):
        """
        Accepts a list of HWC numpy arrays, returns a list of
        lists of label-strings for each image.
        """
        # 1) convert to PIL
        pil_imgs = [Image.fromarray(arr) for arr in img_vectors]

        # 2) tokenize & batch to tensor
        inputs = self.processor(images=pil_imgs, return_tensors="pt").to(self.device)

        # 3) forward
        outputs = self.model(**inputs)

        # 4) post-process: need one target_size per image
        target_sizes = torch.tensor([im.size[::-1] for im in pil_imgs], device=self.device)
        batch_results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )

        # 5) convert to label lists
        all_objs = []
        for result in batch_results:
            labels = [self.model.config.id2label[label_id.item()] 
                      for label_id in result["labels"]]
            all_objs.extend(labels)
        return all_objs