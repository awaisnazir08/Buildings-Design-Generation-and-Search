import os

class MilvusSearch:
    def __init__(self, collection, search_params):
        self.collection = collection
        self.search_params = search_params
    
    def search(self, image_embedding):
        results = self.collection.search(
            data=[image_embedding], 
            anns_field="building_generated_image_embedding", 
            param=self.search_params,
            limit=30,
            expr=None,
            output_fields=['Building_id', "Image_Name"],
            consistency_level="Strong"
        )
        return results

class ImagesRetrieval:
    def __init__(self, root_dir = 'D:\VS Code Folders\Pix2Pix_Buildings\generated_images_512'):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
    
    def get_matched_images_names(self, results):
        images = []
        for result in results[0]:
            images.append(result.entity.get('Image_Name') + '_generated.jpg')
        return images

    def get_images_paths(self, images_names):
        return [os.path.join(self.root_dir, image_name) for image_name in images_names]