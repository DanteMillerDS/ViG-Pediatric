import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import data_loader.extract_data as extract_data
import data_loader.load_data as load_data
import visualize.visualize as visualize
from fine_tune.finetune import TrainModelClassifier
def run_finetune_clip(medical_type, model_type, batch_size):
    """
    Performs fine-tuning of a model on a given medical type and model type.
    :param medical_type: The type of medical data to classify ('ucsd', 'ori').
    :param model_type: The type of model to use for classification ().
    :param batch_size: The batch size for data loading.
    """
    generators, lengths = load_data.create_loader(medical_type, batch_size,finetune = True)
    visualize.save_random_images_from_generators(generators, [medical_type, model_type, "finetune"], 2)
    if model_type == "vig_ti_224_gelu":
        classifier = TrainModelClassifier(medical_type,model_type)
        classifier.run(generators, lengths)
        return classifier
    else:
        print("Did not define a proper classifer!")

if __name__ == "__main__":
    extract_data.mount_and_process()
    batch_size = 128
    model_types = ['vig_ti_224_gelu']
    medical_types = ['ucsd', 'ori']
    ucsd_classifier = run_finetune_clip(medical_types[0], model_types[0], batch_size)
    ori_classifier = run_finetune_clip(medical_types[1], model_types[0], batch_size)