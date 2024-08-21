import clip
from ..utils.helper_utils import load_config
class EmbeddingsModel:
    def __init__(self, device = 'cpu'):
        self.device = device
        self.config = load_config()
        self.clip_model = self.config['model_name']
    
    def load_model(self):
        clip_model, preprocess = clip.load(self.config['model_name'], device=self.device)
        return clip_model, preprocess

if __name__ =='__main__':
    model = EmbeddingsModel()
    print('Clip Loaded...')